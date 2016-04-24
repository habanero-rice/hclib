/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "uts.h"

/***********************************************************
 *                                                         *
 *  Compiler Type (these flags are set by at compile time) *
 *     (default) ANSI C compiler - sequential execution    *
 *     (_OPENMP) OpenMP enabled C compiler                 *
 *     (__UPC__) UPC compiler                              *
 *     (_SHMEM)  Cray Shmem                                *
 *     (__PTHREADS__) Pthreads multithreaded execution     *
 *                                                         *
 ***********************************************************/

#if defined(_OPENMP)
/**** OpenMP Definitions ****/
#include <omp.h>
#define PARALLEL         1
#define COMPILER_TYPE    1
#define SHARED 
#define SHARED_INDEF
#define VOLATILE         volatile
#define MAX_OMP_THREADS       32
#define MAX_SHMEM_THREADS     64
#define LOCK_T           omp_lock_t
#define GET_NUM_THREADS  omp_get_num_threads()
#define GET_THREAD_NUM   omp_get_thread_num()
#define SET_LOCK(zlk)    omp_set_lock(zlk)
#define UNSET_LOCK(zlk)  omp_unset_lock(zlk)
#define INIT_LOCK(zlk)   zlk=omp_global_lock_alloc()
#define INIT_SINGLE_LOCK(zlk) zlk=omp_global_lock_alloc()
#define SMEMCPY          memcpy
#define ALLOC            malloc
#define BARRIER          
// OpenMP helper function to match UPC lock allocation semantics
omp_lock_t * omp_global_lock_alloc() {
  omp_lock_t *lock = (omp_lock_t *) malloc(sizeof(omp_lock_t) + 128);
  omp_init_lock(lock);
  return lock;
}


#else
#error Only supports OMP
#endif /* END Par. Model Definitions */


/***********************************************************
 *  Parallel execution parameters                          *
 ***********************************************************/

int doSteal   = PARALLEL; // 1 => use work stealing
int chunkSize = 20;       // number of nodes to move to/from shared area
int cbint     = 1;        // Cancellable barrier polling interval
int pollint   = 1;        // BUPC Polling interval

size_t n_nodes = 0;
size_t n_leaves = 0;

#ifdef THREAD_METADATA
typedef struct _thread_metadata {
    size_t ntasks;
} thread_metadata;
thread_metadata t_metadata[MAX_OMP_THREADS];
#endif

#define N_BUFFERED_STEALS 16
Node steal_buffer[N_BUFFERED_STEALS];
int n_buffered_steals = 0;
long steal_buffer_locks[MAX_SHMEM_THREADS];

int complete_pes = 0;

static int pe, npes;

static int steal_from(int target_pe, Node *stolen_out) {
    int remote_buffered_steals;

    shmem_set_lock(&steal_buffer_locks[target_pe]);
    shmem_int_get(&remote_buffered_steals, &n_buffered_steals, 1, target_pe);

    if (remote_buffered_steals > 0) {
        remote_buffered_steals--;
        shmem_getmem(stolen_out, &steal_buffer[remote_buffered_steals],
                sizeof(Node), target_pe);
        shmem_int_put(&n_buffered_steals, &remote_buffered_steals, 1, target_pe);
    }

    shmem_clear_lock(&steal_buffer_locks[target_pe]);
    return remote_buffered_steals;
}

static int remote_steal(Node *stolen_out) {
    int pe_above = (pe + 1) % npes;
    int pe_below = pe - 1;
    if (pe_below < 0) pe_below = npes - 1;

    shmem_int_add(&complete_pes, 1, 0);

    int ndone = shmem_int_fadd(&complete_pes, 0, 0);
    while (ndone != npes) {
        // Try to remote steal

        if (steal_from(pe_above, stolen_out) || steal_from(pe_below, stolen_out)) {
            shmem_int_add(&complete_pes, -1, 0);
            return 1;
        }

        pe_above = (pe_above + 1) % npes;
        pe_below = pe_below - 1;
        if (pe_below < 0) pe_below = npes - 1;
        ndone = shmem_int_fadd(&complete_pes, 0, 0);
    }

    assert(ndone == npes);
    return 0;
}

#ifdef __BERKELEY_UPC__
/* BUPC nonblocking I/O Handles */
bupc_handle_t cb_handle       = BUPC_COMPLETE_HANDLE;
const int     local_cb_cancel = 1;
#endif


/***********************************************************
 * Tree statistics (if selected via UTS_STAT)              *
 *   compute overall size and imbalance metrics            *
 *   and histogram size and imbalance per level            *
 ***********************************************************/
#ifdef UTS_STAT

/* Check that we are not being asked to compile parallel with stats.
 * Parallel stats collection is presently not supported.  */
#if PARALLEL
#error "ERROR: Parallel stats collection is not supported!"
#endif

#define MAXHISTSIZE      2000  // max tree depth in histogram
int    stats     = 1;
int    unbType   = 1;
int    maxHeight = 0;         // maximum depth of tree
double maxImb    = 0;         // maximum imbalance
double minImb    = 1;
double treeImb   =-1;         // Overall imbalance, undefined

int    hist[MAXHISTSIZE+1][2];      // average # nodes per level
double unbhist[MAXHISTSIZE+1][3];   // average imbalance per level

int    *rootSize;             // size of the root's children 
double *rootUnb;              // imbalance of root's children

/* Tseng statistics */
int    totalNodes = 0;
double imb_max    = 0;         // % of work in largest child (ranges from 100/n to 100%)
double imb_avg    = 0;
double imb_devmaxavg     = 0;  // ( % of work in largest child ) - ( avg work )
double imb_normdevmaxavg = 0;  // ( % of work in largest child ) - ( avg work ) / ( 100% - avg work )
#else
int stats   = 0;
int unbType = -1;
#endif


/***********************************************************
 *  Execution Tracing                                      *
 ***********************************************************/

#define SS_WORK    0
#define SS_SEARCH  1
#define SS_IDLE    2
#define SS_OVH     3
#define SS_CBOVH   4
#define SS_NSTATES 5

/* session record for session visualization */
struct sessionRecord_t {
  double startTime, endTime;
};
typedef struct sessionRecord_t SessionRecord;

/* steal record for steal visualization */
struct stealRecord_t {
  long int nodeCount;           /* count nodes generated during the session */
  int victimThread;             /* thread from which we stole the work  */
};
typedef struct stealRecord_t StealRecord;

/* Store debugging and trace data */
struct metaData_t {
  SessionRecord sessionRecords[SS_NSTATES][20000];   /* session time records */
  StealRecord stealRecords[20000]; /* steal records */
};
typedef struct metaData_t MetaData;

/* holds text string for debugging info */
char debug_str[1000];


/***********************************************************
 * StealStack types                                        *
 ***********************************************************/

/***********************************************************
 *  Global shared state                                    *
 ***********************************************************/

// termination detection 
VOLATILE SHARED int cb_cancel;
VOLATILE SHARED int cb_count;
VOLATILE SHARED int cb_done;
LOCK_T * cb_lock;

/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// Return a string describing this implementation
char * impl_getName() {
  char * name[] = {"Sequential C", "C/OpenMP", "UPC", "SHMEM", "PThreads"};
  return name[COMPILER_TYPE];
}


// construct string with all parameter settings 
int impl_paramsToStr(char *strBuf, int ind) {
    int n_omp_threads;
#pragma omp parallel
#pragma omp single
    n_omp_threads = omp_get_num_threads();

  ind += sprintf(strBuf+ind, "Execution strategy:  ");
  if (PARALLEL) {
    ind += sprintf(strBuf+ind, "Parallel search using %d threads total (%d "
            "SHMEM PEs, %d OMP threads per PE)\n", npes * n_omp_threads,
            npes, n_omp_threads);
    if (doSteal) {
      ind += sprintf(strBuf+ind, "   Load balance by work stealing, chunk size = %d nodes\n",chunkSize);
      ind += sprintf(strBuf+ind, "  CBarrier Interval: %d\n", cbint);
      ind += sprintf(strBuf+ind, "   Polling Interval: %d\n", pollint);
    }
    else
      ind += sprintf(strBuf+ind, "   No load balancing.\n");
  }
  else
    ind += sprintf(strBuf+ind, "Iterative sequential search\n");
      
  return ind;
}


int impl_parseParam(char *param, char *value) {
  int err = 0;  // Return 0 on a match, nonzero on an error

  switch (param[1]) {
#if (PARALLEL == 1)
    case 'c':
      chunkSize = atoi(value); break;
    case 's':
      doSteal = atoi(value); 
      if (doSteal != 1 && doSteal != 0) 
	err = 1;
      break;
    case 'i':
      cbint = atoi(value); break;
#ifdef __BERKELEY_UPC__
    case 'I':
      pollint = atoi(value); break;
#endif
#else /* !PARALLEL */
#ifdef UTS_STAT
    case 'u':
      unbType = atoi(value); 
      if (unbType > 2) {
        err = 1;
        break;
      }
      if (unbType < 0)
        stats = 0;
      else
        stats = 1;
      break;
#endif
#endif /* PARALLEL */
    default:
      err = 1;
      break;
  }

  return err;
}

void impl_helpMessage() {
  if (PARALLEL) {
    printf("   -s  int   zero/nonzero to disable/enable work stealing\n");
    printf("   -c  int   chunksize for work stealing\n");
    printf("   -i  int   set cancellable barrier polling interval\n");
#ifdef __BERKELEY_UPC__
    printf("   -I  int   set working bupc_poll() interval\n");
#endif
#ifdef __PTHREADS__
    printf("   -T  int   set number of threads\n");
#endif
  } else {
#ifdef UTS_STAT
    printf("   -u  int   unbalance measure (-1: none; 0: min/size; 1: min/n; 2: max/n)\n");
#else
    printf("   none.\n");
#endif
  }
}

void impl_abort(int err) {
#if defined(__UPC__)
  upc_global_exit(err);
#elif defined(_OPENMP)
  exit(err);
#elif defined(_SHMEM)
  exit(err);
#else
  exit(err);
#endif
}


/***********************************************************
 *                                                         *
 *  FUNCTIONS                                              *
 *                                                         *
 ***********************************************************/

/* 
 * StealStack
 *    Stack of nodes with sharing at the bottom of the stack
 *    and exclusive access at the top for the "owning" thread 
 *    which has affinity to the stack's address space.
 *
 *    * All operations on the shared portion of the stack
 *      must be guarded using the stack-specific lock.
 *    * Elements move between the shared and exclusive
 *      portion of the stack solely under control of the 
 *      owning thread. (ss_release and ss_acquire)
 *    * workAvail is the count of elements in the shared
 *      portion of the stack.  It may be read without 
 *      acquiring the stack lock, but of course its value
 *      may not be acurate.  Idle threads read workAvail in
 *      this speculative fashion to minimize overhead to 
 *      working threads.
 *    * Elements can be stolen from the bottom of the shared
 *      portion by non-owning threads.  The values are 
 *      reserved under lock by the stealing thread, and then 
 *      copied without use of the lock (currently space for
 *      reserved values is never reclaimed).
 *
 */

/* fatal error */
void ss_error(char *str) {
  printf("*** [Thread %i] %s\n",GET_THREAD_NUM, str);
  exit(4);
}

#ifdef UTS_STAT
/*
 * Statistics, 
 * : number of nodes per level
 * : imbalanceness of nodes per level
 *
 */
void initHist()
{
  int i;
  for (i=0; i<MAXHISTSIZE; i++){
    hist[i][0]=0;
    hist[i][1]=0;
    unbhist[i][1]=1;
    unbhist[i][2]=0;
  }
}

void updateHist(Node* c, double unb)
{
  if (c->height<MAXHISTSIZE){
    hist[c->height][1]++;
    hist[c->height][0]+=c->numChildren;

    unbhist[c->height][0]+=unb;
    if (unbhist[c->height][1]>unb)
      unbhist[c->height][1]=unb;
    if (unbhist[c->height][2]<unb)
      unbhist[c->height][2]=unb;
		
  }
  else {
    hist[MAXHISTSIZE][1]++;
    hist[MAXHISTSIZE][0]+=c->numChildren;
  }
}

void showHist(FILE *fp)
{
  int i;	
  fprintf(fp, "depth\tavgNumChildren\t\tnumChildren\t imb\t maxImb\t minImb\t\n");
  for (i=0; i<MAXHISTSIZE; i++){
    if ((hist[i][0]!=0)&&(hist[i][1]!=0))
      fprintf(fp, "%d\t%f\t%d\t %lf\t%lf\t%lf\n", i, (double)hist[i][0]/hist[i][1], 
              hist[i][0], unbhist[i][0]/hist[i][1], unbhist[i][1], unbhist[i][2]);	
  }
}

double getImb(Node *c)
{
  int i=0;
  double avg=.0, tmp=.0;
  double unb=0.0;
  
  avg=(double)c->sizeChildren/c->numChildren;

  for (i=0; i<c->numChildren; i++){		
    if ((type==BIN)&&(c->pp==NULL))
      {
        if (unbType<2)
          tmp=min((double)rootSize[i]/avg, avg/(double)rootSize[i]);
        else 
          tmp=max((double)rootSize[i]/avg, avg/(double)rootSize[i]);
        
        if (unbType>0)
          unb+=tmp*rootUnb[i];
        else 
          unb+=tmp*rootUnb[i]*rootSize[i];
      }	
    else{
      if (unbType<2)
        tmp=min((double)c->size[i]/avg, avg/(double)c->size[i]);
      else 
        tmp=max((double)c->size[i]/avg, avg/(double)c->size[i]);
      
      if (unbType>0)
        unb+=tmp*c->unb[i];
      else 
        unb+=tmp*c->unb[i]*c->size[i];
    }
  }
	
  if (unbType>0){
    if (c->numChildren>0) 
      unb=unb/c->numChildren;
    else unb=1.0;
  }
  else {
    if (c->sizeChildren>1) 
      unb=unb/c->sizeChildren;
    else unb=1.0;
  }
  if ((debug & 1) && unb>1) printf("unb>1%lf\t%d\n", unb, c->numChildren);
	
  return unb;
}

void getImb_Tseng(Node *c)
{
  double t_max, t_avg, t_devmaxavg, t_normdevmaxavg;

  if (c->numChildren==0)
    {
      t_avg =0;
      t_max =0;
    }
  else 
    {
      t_max = (double)c->maxSizeChildren/(c->sizeChildren-1);
      t_avg = (double)1/c->numChildren;
    }

  t_devmaxavg = t_max-t_avg;
	
  if (debug & 1)
    printf("max\t%lf, %lf, %d, %d, %d\n", t_max, t_avg, 
           c->maxSizeChildren, c->sizeChildren, c->numChildren);
	
  if (1-t_avg==0)
    t_normdevmaxavg = 1;
  else
    t_normdevmaxavg = (t_max-t_avg)/(1-t_avg);

  imb_max += t_max;
  imb_avg += t_avg;
  imb_devmaxavg += t_devmaxavg;
  imb_normdevmaxavg +=t_normdevmaxavg;
}

void updateParStat(Node *c)
{
  double unb;

  totalNodes++;
  if (maxHeight<c->height) 
    maxHeight=c->height;
	
  unb=getImb(c);
  maxImb=max(unb, maxImb);
  minImb=min(unb, minImb);
  updateHist(c, unb);
  
  getImb_Tseng(c);
	
  if (c->pp!=NULL){
    if ((c->type==BIN)&&(c->pp->pp==NULL)){
      rootSize[c->pp->ind]=c->sizeChildren;
      rootUnb[c->pp->ind]=unb;
    }
    else{
      c->pp->size[c->pp->ind]=c->sizeChildren;
      c->pp->unb[c->pp->ind]=unb;
    }
    /* update statistics per node*/
    c->pp->ind++;
    c->pp->sizeChildren+=c->sizeChildren;
    if (c->pp->maxSizeChildren<c->sizeChildren)
      c->pp->maxSizeChildren=c->sizeChildren;		
  }
  else 
    treeImb = unb;
}
#endif

/*
 *	Tree Implementation      
 *
 */
void initNode(Node * child)
{
  child->type = -1;
  child->height = -1;
  child->numChildren = -1;    // not yet determined

#ifdef UTS_STAT
  if (stats){	
    int i;
    child->ind = 0;
    child->sizeChildren = 1;
    child->maxSizeChildren = 0;
    child->pp = NULL;
    for (i = 0; i < MAXNUMCHILDREN; i++){
      child->size[i] = 0;
      child->unb[i]  = 0.0;
    }
  }
#endif
}


void initRootNode(Node * root, int type)
{
  uts_initRoot(root, type);

  #ifdef TRACE
    stealStack[0]->md->stealRecords[0].victimThread = 0;  // first session is own "parent session"
  #endif

#ifdef UTS_STAT
  if (stats){
    int i;
    root->ind = 0;
    root->sizeChildren = 1;
    root->maxSizeChildren = 1;
    root->pp = NULL;
    
    if (type != BIN){
      for (i=0; i<MAXNUMCHILDREN; i++){
        root->size[i] = 0;
        root->unb[i]  =.0; 
      }
    }
    else {
      int rbf = (int) ceil(b_0);
      rootSize = malloc(rbf*sizeof(int));
      rootUnb = malloc(rbf*sizeof(double));
      for (i = 0; i < rbf; i++) {
        rootSize[i] = 0;
        rootUnb[i] = 0.0;
      }
    }
  }
#endif
}


/* 
 * Generate all children of the parent
 *
 * details depend on tree type, node type and shape function
 *
 */
void genChildren(Node * parent, Node * child) {
  int parentHeight = parent->height;
  int numChildren, childType;

#ifdef THREAD_METADATA
  t_metadata[omp_get_thread_num()].ntasks += 1;
#endif

#pragma omp atomic
  n_nodes += 1;

  numChildren = uts_numChildren(parent);
  childType   = uts_childType(parent);

  // fprintf(stderr, "numChildren = %d\n", numChildren);

  // record number of children in parent
  parent->numChildren = numChildren;
  
  // construct children and push onto stack
  if (numChildren > 0) {
    int i, j;
    child->type = childType;
    child->height = parentHeight + 1;

#ifdef UTS_STAT
    if (stats) {
      child->pp = parent;  // pointer to parent
    }
#endif

    unsigned char * parent_state = parent->state.state;
    unsigned char * child_state = child->state.state;

    for (i = 0; i < numChildren; i++) {
      for (j = 0; j < computeGranularity; j++) {
        // TBD:  add parent height to spawn
        // computeGranularity controls number of rng_spawn calls per node
          rng_spawn(parent_state, child_state, i);
      }

      Node parent = *child;

      if (omp_get_thread_num() == 0 && n_buffered_steals < N_BUFFERED_STEALS) {
          steal_buffer[n_buffered_steals++] = parent;
      } else {
#pragma omp task untied firstprivate(parent)
          {
              Node child;
              initNode(&child);

              if (parent.numChildren < 0) {
                  genChildren(&parent, &child);
              }
          }
      }
    }
  } else {
#pragma omp atomic
      n_leaves += 1;
  }
}

// causes one or more threads waiting at barrier, if any,
//  to be released
#ifdef TRACE
// print session records for each thread (used when trace is enabled)
void printSessionRecords()
{
  int i, j, k;
  double offset;

  for (i = 0; i < GET_NUM_THREADS; i++) {
    offset = startTime[i] - startTime[0];

    for (j = 0; j < SS_NSTATES; j++)
       for (k = 0; k < stealStack[i]->entries[j]; k++) {
          printf ("%d %d %f %f", i, j,
            stealStack[i]->md->sessionRecords[j][k].startTime - offset,
            stealStack[i]->md->sessionRecords[j][k].endTime - offset);
          if (j == SS_WORK)
            printf (" %d %ld",
              stealStack[i]->md->stealRecords[k].victimThread,
              stealStack[i]->md->stealRecords[k].nodeCount);
            printf ("\n");
     }
  }
}
#endif

// display search statistics
void showStats(double elapsedSecs) {
  int i;
  int tnodes = 0, tleaves = 0, trel = 0, tacq = 0, tsteal = 0, tfail= 0;
  int mdepth = 0, mheight = 0;
  double twork = 0.0, tsearch = 0.0, tidle = 0.0, tovh = 0.0, tcbovh = 0.0;

//   // combine measurements from all threads
//   for (i = 0; i < GET_NUM_THREADS; i++) {
//     tnodes  += stealStack[i]->nNodes;
//     tleaves += stealStack[i]->nLeaves;
//     trel    += stealStack[i]->nRelease;
//     tacq    += stealStack[i]->nAcquire;
//     tsteal  += stealStack[i]->nSteal;
//     tfail   += stealStack[i]->nFail;
//     twork   += stealStack[i]->time[SS_WORK];
//     tsearch += stealStack[i]->time[SS_SEARCH];
//     tidle   += stealStack[i]->time[SS_IDLE];
//     tovh    += stealStack[i]->time[SS_OVH];
//     tcbovh  += stealStack[i]->time[SS_CBOVH];
//     mdepth   = max(mdepth, stealStack[i]->maxStackDepth);
//     mheight  = max(mheight, stealStack[i]->maxTreeDepth);
//   }
//   if (trel != tacq + tsteal) {
//     printf("*** error! total released != total acquired + total stolen\n");
//   }
//     
  uts_showStats(GET_NUM_THREADS, chunkSize, elapsedSecs, n_nodes, n_leaves, mheight);
// 
//   if (verbose > 1) {
//     if (doSteal) {
//       printf("Total chunks released = %d, of which %d reacquired and %d stolen\n",
//           trel, tacq, tsteal);
//       printf("Failed steal operations = %d, ", tfail);
//     }
// 
//     printf("Max stealStack size = %d\n", mdepth);
//     printf("Avg time per thread: Work = %.6f, Search = %.6f, Idle = %.6f\n", (twork / GET_NUM_THREADS),
//         (tsearch / GET_NUM_THREADS), (tidle / GET_NUM_THREADS));
//     printf("                     Overhead = %6f, CB_Overhead = %6f\n\n", (tovh / GET_NUM_THREADS),
//         (tcbovh/GET_NUM_THREADS));
//   }
// 
//   // per thread execution info
//   if (verbose > 2) {
//     for (i = 0; i < GET_NUM_THREADS; i++) {
//       printf("** Thread %d\n", i);
//       printf("  # nodes explored    = %d\n", stealStack[i]->nNodes);
//       printf("  # chunks released   = %d\n", stealStack[i]->nRelease);
//       printf("  # chunks reacquired = %d\n", stealStack[i]->nAcquire);
//       printf("  # chunks stolen     = %d\n", stealStack[i]->nSteal);
//       printf("  # failed steals     = %d\n", stealStack[i]->nFail);
//       printf("  maximum stack depth = %d\n", stealStack[i]->maxStackDepth);
//       printf("  work time           = %.6f secs (%d sessions)\n",
//              stealStack[i]->time[SS_WORK], stealStack[i]->entries[SS_WORK]);
//       printf("  overhead time       = %.6f secs (%d sessions)\n",
//              stealStack[i]->time[SS_OVH], stealStack[i]->entries[SS_OVH]);
//       printf("  search time         = %.6f secs (%d sessions)\n",
//              stealStack[i]->time[SS_SEARCH], stealStack[i]->entries[SS_SEARCH]);
//       printf("  idle time           = %.6f secs (%d sessions)\n",
//              stealStack[i]->time[SS_IDLE], stealStack[i]->entries[SS_IDLE]);
//       printf("  wakeups             = %d, false wakeups = %d (%.2f%%)",
//              stealStack[i]->wakeups, stealStack[i]->falseWakeups,
//              (stealStack[i]->wakeups == 0) ? 0.00 : ((((double)stealStack[i]->falseWakeups)/stealStack[i]->wakeups)*100.0));
//       printf("\n");
//     }
//   }
// 
//   #ifdef TRACE
//     printSessionRecords();
//   #endif
// 
//   // tree statistics output to stat.txt, if requested
// #ifdef UTS_STAT
//   if (stats) {
//     FILE *fp;
//     char * tmpstr;
//     char strBuf[5000];
//     int  ind = 0;
//     
//     fp = fopen("stat.txt", "a+w");
//     fprintf(fp, "\n------------------------------------------------------------------------------------------------------\n");
//     ind = uts_paramsToStr(strBuf, ind);
//     ind = impl_paramsToStr(strBuf, ind);
//     //showParametersStr(strBuf);
//     fprintf(fp, "%s\n", strBuf);
//     
//     fprintf(fp, "\nTotal nodes = %d\n", totalNodes); 
//     fprintf(fp, "Max depth   = %d\n\n", maxHeight); 
//     fprintf(fp, "Tseng ImbMeasure(overall)\n max:\t\t%lf \n avg:\t\t%lf \n devMaxAvg:\t %lf\n normDevMaxAvg: %lf\t\t\n\n", 
//             imb_max/totalNodes, imb_avg/totalNodes, imb_devmaxavg/totalNodes, 
//             imb_normdevmaxavg/totalNodes);
//     
//     switch (unbType){
//     case 0: tmpstr = "(min imb weighted by size)"; break;
//     case 1: tmpstr = "(min imb not weighted by size)"; break;
//     case 2: tmpstr = "(max imb not weighted by size)"; break;
//     default: tmpstr = "(?unknown measure)"; break;
//     }
//     fprintf(fp, "ImbMeasure:\t%s\n Overall:\t %lf\n Max:\t\t%lf\n Min:\t\t%lf\n\n", 
//             tmpstr, treeImb, minImb, maxImb);
//     showHist(fp);
//     fprintf(fp, "\n------------------------------------------------------------------------------------------------------\n\n\n");
//     fclose(fp);
//   }
// #endif
}


/*  Main() function for: Sequential, OpenMP, UPC, and Shmem
 *
 *  Notes on execution model:
 *     - under openMP, global vars are all shared
 *     - under UPC, global vars are private unless explicitly shared
 *     - UPC is SPMD starting with main, OpenMP goes SPMD after
 *       parsing parameters
 */
int main(int argc, char *argv[]) {
  Node root;

#ifdef THREAD_METADATA
  memset(t_metadata, 0x00, MAX_OMP_THREADS * sizeof(thread_metadata));
#endif
  memset(steal_buffer_locks, 0x00, MAX_SHMEM_THREADS * sizeof(long));

  shmem_init();

  pe = shmem_my_pe();
  npes = shmem_n_pes();

  /* determine benchmark parameters (all PEs) */
  uts_parseParams(argc, argv);

#ifdef UTS_STAT
  if (stats) {
    initHist();
  }
#endif  

  double t1, t2, et;

  /* show parameter settings */
  if (pe == 0) {
      uts_printParams();
  }

  initRootNode(&root, type);

  shmem_barrier_all();

  /* time parallel search */
  t1 = uts_wctime();

  int n_omp_threads;

/********** SPMD Parallel Region **********/
#pragma omp parallel
  {
#pragma omp single
      {
          int first = 1;
          n_omp_threads = omp_get_num_threads();
          assert(n_omp_threads <= MAX_OMP_THREADS);

          Node child;
retry:
          initNode(&child);

          if (first) {
              if (pe == 0) {
                  genChildren(&root, &child);
              }
          } else {
              genChildren(&root, &child);
          }
          first = 0;

#pragma omp taskwait

          if (n_buffered_steals > 0) {
              shmem_set_lock(&steal_buffer_locks[pe]);
              if (n_buffered_steals > 0) {
                  n_buffered_steals--;
                  memcpy(&root, &steal_buffer[n_buffered_steals], sizeof(root));
                  shmem_clear_lock(&steal_buffer_locks[pe]);
                  goto retry;
              } else {
                  shmem_clear_lock(&steal_buffer_locks[pe]);
              }
          }

          const int got_more_work = remote_steal(&root);
          if (got_more_work) {
              goto retry;
          }
      }
  }

  if (pe != 0) {
      shmem_int_add(&n_nodes, n_nodes, 0);
      shmem_int_add(&n_leaves, n_leaves, 0);
  }

  shmem_barrier_all();

  t2 = uts_wctime();
  et = t2 - t1;
  if (pe == 0) {
      showStats(et);
  }
/********** End Parallel Region **********/
#ifdef THREAD_METADATA
  int p;
  for (p = 0; p < npes; p++) {
      if (p == pe) {
          printf("\n");
          int i;
          for (i = 0; i < n_omp_threads; i++) {
              printf("PE %d, thread %d: %lu tasks\n", p, i, t_metadata[i].ntasks);
          }
      }
      shmem_barrier_all();
  }
#endif

  shmem_finalize();
  return 0;
}
