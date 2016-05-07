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
 *  One each node, a set of MAX_NUM_THREADS steal stacks is allocated in the
 *  symmetric heap. The shmem_my_pe()-th member of this set is the stack
 *  associated with this PE, which is primarily a set of node slots (Node) that
 *  is the backing data of the stack with length MAXSTACKDEPTH and a top pointer
 *  that is the index of the lowest empty slot in the stack. The stack
 *  for PE 0 is initialized with a single Node before execution really starts.
 *
 *  Each stack for each PE is separated in to three sections: the local section,
 *  the non-local and non-stolen section, and the non-local and stolen section.
 *  The non-local and stolen section is at the base, from index 0 to
 *  ss.sharedStart. This region represents the locally created nodes that have
 *  been stolen from the current PE by another PE. The non-local and non-stolen
 *  section goes from ss.sharedStart to ss.local. This section contains nodes
 *  that were locally created and are eligible for stealing, but have not been
 *  stolen yet. These nodes may also be reclaimed in to the local section by
 *  this PE if it needs more work. The local section is above that, from
 *  ss.local to ss.top and contains locally created work that only this PE has
 *  access to. If work-stealing is enabled and the size of the local section is
 *  too big (as defined by chunkSize), then the local region is reduced by
 *  incrementing ss.local to allow other PEs to steal work from this PE.
 *
 *  During the parallel tree search, we loop until there are no more local nodes
 *  left. Every time we visit a node we generate a certain number of children
 *  for that node and push each of them on to the stack. Once we run out of
 *  local work, we try to acquire locally-created work from the non-local region
 *  by decrementing local (ss_acquire). If that fails, we then must try to find
 *  a remote PE to steal from and grab a chunk of nodes from it using OpenSHMEM
 *  copies (ss_steal) and placing them on the local stack of our PE. If this
 *  succeeds, we loop back around to processing these nodes.
 *
 *  If we are unable to find work locally or remotely, we enter a cancellable
 *  barrier that waits for all PEs to enter it. This barrier is cancelled if any
 *  PEs release nodes for stealing. If the barrier completes successfully with
 *  all nodes having entered it, the search exits.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <shmem.h>
#include <omp.h>
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

omp_lock_t * omp_global_lock_alloc() {
  omp_lock_t *lock = (omp_lock_t *) malloc(sizeof(omp_lock_t) + 128);
  omp_init_lock(lock);
  return lock;
}

long * smem_global_lock_alloc() {    
    long *lock = (long *) shmem_malloc(sizeof(long));
    *lock = 0;
    shmem_barrier_all();
    return lock;
}

#define MAX_OMP_THREADS       32
#define MAX_SHMEM_THREADS       64

#define PUT_ALL(a,b)								\
	do {						 			\
		int _iter, _node; 						\
		for (_iter = 1; _iter < shmem_n_pes(); _iter++) {		\
			_node = (shmem_my_pe() + _iter) % shmem_n_pes();	\
			shmem_int_put((int *)&a,(int *)&b,1,_node); \
		}								\
	} while(0)

/***********************************************************
 *  Parallel execution parameters                          *
 ***********************************************************/

int chunkSize = 20;       // number of nodes to move to/from shared area
int cbint     = 1;        // Cancellable barrier polling interval
int pollint   = 1;        // BUPC Polling interval

/***********************************************************
 * Tree statistics (if selected via UTS_STAT)              *
 *   compute overall size and imbalance metrics            *
 *   and histogram size and imbalance per level            *
 ***********************************************************/
int stats   = 0;
int unbType = -1;

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

#define MAXSTACKDEPTH 500000

/* stack of nodes */
struct stealStack_t
{
  int stackSize;     /* total space avail (in number of elements) */
  int workAvail;     /* elements available for stealing */
  int sharedStart;   /* index of start of shared portion of stack */
  int local;         /* index of start of local portion */
  int top;           /* index of stack top */
  int maxStackDepth;                      /* stack stats */ 
  int nNodes, maxTreeDepth;               /* tree stats  */
  int nLeaves;
  int nAcquire, nRelease, nSteal, nFail;  /* steal stats */
  int nSuccessfulRemoteAcquire;
  int nFailedRemoteAcquire;
  int nSuccessfulRemoteSteal;
  int nFailedRemoteSteal;
  double timeInWait;
  int nWaits;
  double timeInSharedSpin;
  int nSharedSpins;

  int wakeups, falseWakeups, nNodes_last;
  double time[SS_NSTATES], timeLast;         /* perf measurements */
  int entries[SS_NSTATES], curState; 
  long * remote_stackLock;
  omp_lock_t * local_stackLock;
  Node * stack;       /* addr of actual stack of nodes in local addr space */
  Node * stack_g; /* addr of same stack in global addr space */
};
typedef struct stealStack_t StealStack;

typedef StealStack * SharedStealStackPtr;


/***********************************************************
 *  Global shared state                                    *
 ***********************************************************/

// shared access to each thread's stealStack
SharedStealStackPtr local_stealStack[MAX_OMP_THREADS];
SharedStealStackPtr remote_stealStack[MAX_SHMEM_THREADS];

// termination detection 
volatile int local_cb_cancel;
volatile int local_cb_count;
volatile int local_cb_done;
omp_lock_t * local_cb_lock;

volatile int shared_cb_cancel;
volatile int shared_cb_count;
volatile int shared_cb_done;
long * shared_cb_lock;

static int num_omp_threads = -1;

/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// Return a string describing this implementation
char * impl_getName() {
  return "SHMEM-OMP";
}

// construct string with all parameter settings 
int impl_paramsToStr(char *strBuf, int ind) {
  ind += sprintf(strBuf+ind, "Execution strategy:  ");
  ind += sprintf(strBuf+ind, "Parallel search using %d OMP threads, %d PEs\n", num_omp_threads,
          shmem_n_pes());
  ind += sprintf(strBuf+ind, "   Load balance by work stealing, chunk size = %d nodes\n", chunkSize);
  ind += sprintf(strBuf+ind, "  CBarrier Interval: %d\n", cbint);
  ind += sprintf(strBuf+ind, "   Polling Interval: %d\n", pollint);
      
  return ind;
}

int impl_parseParam(char *param, char *value) {
  int err = 0;  // Return 0 on a match, nonzero on an error

  switch (param[1]) {
    case 'c':
      chunkSize = atoi(value); break;
    case 'i':
      cbint = atoi(value); break;
    default:
      err = 1;
      break;
  }

  return err;
}

void impl_helpMessage() {
  printf("   -s  int   zero/nonzero to disable/enable work stealing\n");
  printf("   -c  int   chunksize for work stealing\n");
  printf("   -i  int   set cancellable barrier polling interval\n");
}

void impl_abort(int err) {
  exit(err);
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

/* restore stack to empty state */
void ss_mkEmpty(StealStack *s) {
  s->sharedStart = 0;
  s->local  = 0;
  s->top    = 0;
  s->workAvail = 0;
}

/* fatal error */
void ss_error(char *str) {
  fprintf(stderr, "*** [Thread %i on PE %i] %s\n", omp_get_thread_num(), shmem_my_pe(),
          str);
  exit(42);
}

/* initialize the stack */
void ss_init(StealStack *s, int nelts) {

  if (debug & 1)
    printf("Thread %d on PE %d intializing stealStack %p, sizeof(Node) = %X\n", 
           omp_get_thread_num(), shmem_my_pe(), s, (int)(sizeof(Node)));

  s->stackSize = nelts;
  s->nNodes = 0;
  s->maxStackDepth = 0;
  s->maxTreeDepth = 0;
  s->nLeaves = 0;
  s->nAcquire = 0;
  s->nSuccessfulRemoteAcquire = 0;
  s->nFailedRemoteAcquire = 0;
  s->nSuccessfulRemoteSteal = 0;
  s->nFailedRemoteSteal = 0;
  s->timeInWait = 0.0;
  s->nWaits = 0;
  s->timeInSharedSpin = 0;
  s->nSharedSpins = 0;
  s->nRelease = 0;
  s->nSteal = 0;
  s->nFail = 0;
  s->wakeups = 0;
  s->falseWakeups = 0;
  s->nNodes_last = 0;
  ss_mkEmpty(s);
}


/* local push */
void ss_push(StealStack *s, Node *c) {
  if (s->top >= s->stackSize)
    ss_error("ss_push: overflow");
  memcpy(&(s->stack[s->top]), c, sizeof(Node));
  s->top++;
  s->nNodes++;
  s->maxStackDepth = max(s->top, s->maxStackDepth);
  s->maxTreeDepth = max(s->maxTreeDepth, c->height);
}

/* local top:  get local addr of node at top */ 
Node * ss_top(StealStack *s) {
  Node *r;
  if (s->top <= s->local)
    ss_error("ss_top: empty local stack");
  r = &(s->stack[(s->top) - 1]);
  return r;
}

/* local pop */
void ss_pop(StealStack *s) {
  Node *r;
  if (s->top <= s->local)
    ss_error("ss_pop: empty local stack");
  s->top--;
  r = &(s->stack[s->top]);
}
  
/* local top position:  stack index of top element */
int ss_topPosn(StealStack *s)
{
  if (s->top <= s->local)
    ss_error("ss_topPosn: empty local stack");
  return s->top - 1;
}

/* local depth */
int ss_localDepth(StealStack *s) {
  return (s->top - s->local);
}

/* release k values from bottom of local stack */
void ss_release_local(StealStack *s, int k) {
  omp_set_lock(s->local_stackLock);
  if (s->top - s->local >= k) {
    s->local += k;
    s->workAvail += k;
    s->nRelease++;
    // fprintf(stderr, "%d %d releasing %d local, top=%d local=%d sharedStart=%d workAvail=%d\n", shmem_my_pe(), omp_get_thread_num(), k, s->top, s->local, s->sharedStart, s->workAvail);
  }
  else
  {
    ss_error("ss_release:  do not have k vals to release");
  }
  omp_unset_lock(s->local_stackLock);
}

void ss_release_remote(StealStack *local_ss, StealStack *remote_ss, int k) {
    assert(omp_get_thread_num() == 0);
    assert(remote_ss->remote_stackLock);
    assert(local_ss->local_stackLock);

    shmem_set_lock(remote_ss->remote_stackLock);
    omp_set_lock(local_ss->local_stackLock);

    if (local_ss->workAvail >= k) {
        memcpy(remote_ss->stack_g + remote_ss->top,
                local_ss->stack_g + local_ss->sharedStart, k * sizeof(Node));
        remote_ss->nSteal++;
        remote_ss->top += k;
        remote_ss->local += k;
        remote_ss->workAvail += k;
        remote_ss->nRelease++;

        local_ss->sharedStart += k;
        local_ss->workAvail -= k;
    // fprintf(stderr, "%d %d releasing %d remote, remote_ss->top=%d remote_ss->local=%d remote_ss->sharedStart=%d remote_ss->workAvail=%d\n", shmem_my_pe(), omp_get_thread_num(), k, remote_ss->top, remote_ss->local, remote_ss->sharedStart, remote_ss->workAvail);
    }

    omp_unset_lock(local_ss->local_stackLock);
    shmem_clear_lock(remote_ss->remote_stackLock);
}

/* move k values from top of shared stack into local stack
 * return false if k vals are not avail on shared stack
 */
int ss_acquire(StealStack *s, int k) {
  int avail;
  omp_set_lock(s->local_stackLock);
  avail = s->local - s->sharedStart;
  if (avail >= k) {
    s->local -= k;
    s->workAvail -= k;
    s->nAcquire++;
  }
  omp_unset_lock(s->local_stackLock);
  return (avail >= k);
}

int ss_remote_acquire(StealStack *remote_ss, StealStack *local_ss, int k) {
    int success = 0;
    assert(omp_get_thread_num() == 0);
    if (remote_ss->workAvail < k) {
        return 0;
    }

    shmem_set_lock(remote_ss->remote_stackLock);
    omp_set_lock(local_ss->local_stackLock);

    if (remote_ss->workAvail >= k) {
        memcpy(local_ss->stack + local_ss->top,
                remote_ss->stack_g + remote_ss->sharedStart, k * sizeof(Node));
        local_ss->nSuccessfulRemoteAcquire++;
        local_ss->nSteal++;
        local_ss->top += k;

        remote_ss->sharedStart += k;
        remote_ss->workAvail -= k;
        success = 1;
    } else {
        local_ss->nFailedRemoteAcquire++;
    }

    omp_unset_lock(local_ss->local_stackLock);
    shmem_clear_lock(remote_ss->remote_stackLock);
    return success;
}

/* steal k values from shared portion of victim thread's stealStack
 * onto local portion of current thread's stealStack.
 * return false if k vals are not avail in victim thread
 */
static int ss_steal(StealStack *this_ss, StealStack *victim_ss, int victim, int k,
        int is_local) {
  if (!is_local) {
      assert(omp_get_thread_num() == 0);
  }

  int victimLocal, victimShared, victimWorkAvail;
  int ok;

  // fprintf(stderr, "%d %d trying to steal from victim %d is_local=%d, stack %p, "
  //         "sharedStart=%d top=%d workAvail=%d\n",
  //         shmem_my_pe(), omp_get_thread_num(),
  //         victim, is_local, victim_ss, this_ss->sharedStart, this_ss->top, this_ss->workAvail);
 
  if (this_ss->sharedStart != this_ss->top) {
    ss_error("ss_steal: thief attempts to steal onto non-empty stack");
  }

  if (this_ss->top + k >= this_ss->stackSize) {
    ss_error("ss_steal: steal will overflow thief's stack");
  }
 
  if (is_local) {
      omp_set_lock(victim_ss->local_stackLock);
  } else {
      // fprintf(stderr, "%d Setting lock %p %p\n", shmem_my_pe(), victim_ss, victim_ss->remote_stackLock);
      shmem_set_lock(victim_ss->remote_stackLock);
      // fprintf(stderr, "%d lock set %p %p\n", shmem_my_pe(), victim_ss, victim_ss->remote_stackLock);
  }
 
  if (!is_local) {
      /* Get remote steal stack */
      shmem_getmem(victim_ss, victim_ss, sizeof(StealStack)-4*sizeof(void*),
              victim);
  }

  victimLocal = victim_ss->local;
  victimShared = victim_ss->sharedStart;
  victimWorkAvail = victim_ss->workAvail;
  
  if (victimLocal - victimShared != victimWorkAvail) {
    ss_error("ss_steal: stealStack invariant violated");
  }
  
  ok = victimWorkAvail >= k;
  if (ok) {
    /* reserve a chunk */
    victim_ss->sharedStart =  victimShared + k;
    victim_ss->workAvail = victimWorkAvail - k;

    if (!is_local) {
        // FIXME: These transfers ought to be combined.  They can't be
        // though because the data protected by the stacklock is not
        // the only data in the StealStack structure.
        shmem_int_put(&(victim_ss->sharedStart), &(victim_ss->sharedStart), 1, victim);
        shmem_int_put(&(victim_ss->workAvail), &(victim_ss->workAvail), 1, victim);
    }
  }

  if (is_local) {
      omp_unset_lock(victim_ss->local_stackLock);
  } else {
      // fprintf(stderr, "%d Clearing lock %p %p\n", shmem_my_pe(), victim_ss, victim_ss->remote_stackLock);
      shmem_clear_lock(victim_ss->remote_stackLock);
  }
  
  /* if k elts reserved, move them to local portion of our stack */
  if (ok) {
    Node * victimStackBase = victim_ss->stack_g;
    Node * victimSharedStart = victimStackBase + victimShared;

    if (is_local) {
        memcpy(&(this_ss->stack[this_ss->top]), victimSharedStart,
                k * sizeof(Node));
    } else {
        shmem_getmem(&(this_ss->stack[this_ss->top]), victimSharedStart,
                k * sizeof(Node), victim);
    }

    this_ss->nSteal++;
    this_ss->top += k;
    if (!is_local) {
        this_ss->local += k;
        this_ss->workAvail += k;
    }
  }
  else {
    this_ss->nFail++;
  }

  if (!is_local) {
      if (ok) {
          this_ss->nSuccessfulRemoteSteal++;
      } else {
          this_ss->nFailedRemoteSteal++;
      }
  }

  return (ok);
}

/* search other threads for work to steal */
int findwork_local(int k) {
    const int curr_thread = omp_get_thread_num();
    const int num_threads = omp_get_num_threads();

    int i,v;
    for (i = 1; i < num_threads; i++) {
        v = (curr_thread + i) % num_threads;
        if (local_stealStack[v]->workAvail >= k)
            return v;
    }
    return -1;
}

int findwork_remote(int k) {
    assert(omp_get_thread_num() == 0);
    const int curr_thread = shmem_my_pe();
    const int num_threads = shmem_n_pes();

    int i,v;
    for (i = 1; i < num_threads; i++) {
        v = (curr_thread + i) % num_threads;
        shmem_int_get(&(remote_stealStack[v]->workAvail),
                &(remote_stealStack[v]->workAvail), 1, v);
        if (remote_stealStack[v]->workAvail >= k) {
            return v;
        }
    }
    return -1;
}

/**
 *  Tracing functions
 *   Track changes in the search state for offline analysis.
**/
void ss_initState(StealStack *s) {
  int i;
  s->timeLast = uts_wctime();
  for (i = 0; i < SS_NSTATES; i++) {
    s->time[i] = 0.0;
    s->entries[i] = 0;
  }
  s->curState = SS_IDLE;
}

void ss_setState(StealStack *s, int state){
  double time;
  if (state < 0 || state >= SS_NSTATES)
    ss_error("ss_setState: thread state out of range");
  if (state == s->curState)
    return;
  time = uts_wctime();
  s->time[s->curState] +=  time - s->timeLast;

  s->entries[state]++;
  s->timeLast = time;
  s->curState = state;
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


// forward decl
void releaseNodes(StealStack *local_ss, StealStack *remote_ss);

/* 
 * Generate all children of the parent
 *
 * details depend on tree type, node type and shape function
 *
 */
void genChildren(Node * parent, Node * child, StealStack * local_ss,
        StealStack * remote_ss) {
  int parentHeight = parent->height;
  int numChildren, childType;

  numChildren = uts_numChildren(parent);
  childType   = uts_childType(parent);

  // record number of children in parent
  parent->numChildren = numChildren;
  
  // construct children and push onto stack
  if (numChildren > 0) {
    int i, j;
    child->type = childType;
    child->height = parentHeight + 1;

#ifdef UTS_STAT
    if (stats)
      child->pp = parent;  // pointer to parent
#endif

    for (i = 0; i < numChildren; i++) {
      for (j = 0; j < computeGranularity; j++) {
        // TBD:  add parent height to spawn
        // computeGranularity controls number of rng_spawn calls per node
        rng_spawn(parent->state.state, child->state.state, i);
      }
      ss_push(local_ss, child);
      releaseNodes(local_ss, remote_ss);
    }
  } else {
    local_ss->nLeaves++;
  }
}


    
/*
 *  Parallel tree traversal
 *
 */

// cancellable barrier

// initialize lock:  single thread under omp, all threads under upc
void cb_init(){

    local_cb_lock = omp_global_lock_alloc();
    shared_cb_lock = smem_global_lock_alloc();

    shmem_set_lock(shared_cb_lock);
    local_cb_count = 0;
    local_cb_cancel = 0;
    local_cb_done = 0;
    shared_cb_count = 0;
    shared_cb_cancel = 0;
    shared_cb_done = 0;
    shmem_clear_lock(shared_cb_lock);
}

//  delay this thread until all threads arrive at barrier
//     or until barrier is cancelled
int cbarrier_wait(StealStack *local_ss) {
    // fprintf(stderr, "Thread %d on PE %d waiting\n", omp_get_thread_num(), shmem_my_pe());
  int local_l_count, local_l_done, local_l_cancel;
  int shared_l_count, shared_l_done, shared_l_cancel;

  const int thread = omp_get_thread_num();
  const int pe = shmem_my_pe();

  const double startTime = uts_wctime();

  omp_set_lock(local_cb_lock);
  local_cb_count++;
  local_l_count = local_cb_count;
  local_l_done = local_cb_done;
  omp_unset_lock(local_cb_lock);

  if (omp_get_thread_num() == 0) {
      // Wait for all threads to enter local barrier
      do {
          local_l_count = local_cb_count;
          local_l_cancel = local_cb_cancel;
      } while (!local_l_cancel && local_l_count < omp_get_num_threads());

      if (!local_l_cancel) {
          shmem_set_lock(shared_cb_lock);
          shared_cb_count++;
          PUT_ALL(shared_cb_count, shared_cb_count);
          if (shared_cb_count == shmem_n_pes()) {
              shared_cb_done = 1;
              PUT_ALL(shared_cb_done, shared_cb_done);
          }
          shared_l_count = shared_cb_count;
          shared_l_done = shared_cb_done;
          // fprintf(stderr, "%d %d: shared_l_done=%d shared_cb_count=%d\n", shmem_my_pe(), omp_get_thread_num(), shared_l_done, shared_cb_count);
          shmem_clear_lock(shared_cb_lock);

          const double startSharedSpin = uts_wctime();
          do {
              shared_l_count = shared_cb_count;
              shared_l_cancel = shared_cb_cancel;
              shared_l_done = shared_cb_done;
              // fprintf(stderr, "  %d %d: shared_l_cancel = %d shared_l_done = %d\n", shmem_my_pe(), omp_get_thread_num(), shared_l_cancel, shared_l_done);
          }
          while (!shared_l_cancel && !shared_l_done);
          const double endSharedSpin = uts_wctime();
          local_ss->timeInSharedSpin += (endSharedSpin - startSharedSpin);
          local_ss->nSharedSpins++;

          shmem_set_lock(shared_cb_lock);
          shared_cb_count--;
          shared_l_count = shared_cb_count;
          PUT_ALL(shared_cb_count, shared_cb_count);
          shared_cb_cancel = 0;
          shared_l_done = shared_cb_done;
          shmem_clear_lock(shared_cb_lock);

          if (shared_l_cancel) {
              local_l_cancel;
          } else {
              assert(shared_l_done);
              local_l_done = shared_l_done;
          }
      }

      omp_set_lock(local_cb_lock);
      // fprintf(stderr, "%d: at end shared_cb_done=%d\n", omp_get_thread_num(), shared_cb_done);
      if (shared_cb_done) {
          local_cb_done = 1;
      }
      local_cb_count--;
      local_l_count = local_cb_count;
      local_cb_cancel = 0;
      local_l_done = local_cb_done;
      omp_unset_lock(local_cb_lock);

  } else {
      // spin locally if we are not the master thread
      do {
          local_l_count = local_cb_count;
          local_l_cancel = local_cb_cancel;
          local_l_done = local_cb_done;
      }
      while (!local_l_cancel && !local_l_done);

      // fprintf(stderr, "%d: out of spin loop local_l_cancel=%d local_l_done=%d\n", omp_get_thread_num(), local_l_cancel, local_l_done);

      omp_set_lock(local_cb_lock);
      local_cb_count--;
      local_l_count = local_cb_count;
      local_cb_cancel = 0;
      local_l_done = local_cb_done;
      omp_unset_lock(local_cb_lock);
  }

  const double endTime = uts_wctime();
  local_ss->timeInWait += (endTime - startTime);
  local_ss->nWaits++;

  return local_cb_done;
}

// causes one or more threads waiting at barrier, if any,
//  to be released
void remote_cbarrier_cancel() {
    shared_cb_cancel = 1;
    PUT_ALL(shared_cb_cancel, shared_cb_cancel);
}

void local_cbarrier_cancel() {
    local_cb_cancel = 1;
}

void releaseNodes(StealStack *local_ss, StealStack *remote_ss) {
    // fprintf(stderr, "%d: releasing nodes with local depth %d, chunkSize=%d, "
    //         "workAvail=%d\n", omp_get_thread_num(), ss_localDepth(local_ss),
    //         chunkSize, local_ss->workAvail);
  if (ss_localDepth(local_ss) > 2 * chunkSize) {
    // Attribute this time to runtime overhead
    ss_setState(local_ss, SS_OVH);
    if (omp_get_thread_num() == 0 && local_ss->workAvail > 2 * chunkSize) {
        // Release remotely
        ss_release_remote(local_ss, remote_ss, chunkSize);
        // This has significant overhead on clusters!
        if (local_ss->nNodes % cbint == 0) {
          ss_setState(local_ss, SS_CBOVH);
          remote_cbarrier_cancel();
        }
    } else {
        // Release locally
        ss_release_local(local_ss, chunkSize);
        if (local_ss->nNodes % cbint == 0) {
          ss_setState(local_ss, SS_CBOVH);
          local_cbarrier_cancel();
        }
    }

    ss_setState(local_ss, SS_WORK);
  }
}

/* 
 * parallel search of UTS trees using work stealing 
 * 
 *   Note: tree size is measured by the number of
 *         push operations
 */
void parTreeSearch(StealStack *local_ss, StealStack *remote_ss) { // TODO
  int done = 0;
  Node * parent;
  Node child;

  /* template for children */
  initNode(&child);

  /* tree search */
  while (done == 0) {
    
    /* local work */
    while (ss_localDepth(local_ss) > 0) {		

      ss_setState(local_ss, SS_WORK);

      /* examine node at stack top */
      parent = ss_top(local_ss);
      if (parent->numChildren < 0) {
          // first time visited, construct children and place on stack
          genChildren(parent, &child, local_ss, remote_ss);
      }
      else {
	// second time visit, process accumulated statistics and pop
#ifdef UTS_STAT
        if (stats)
          updateParStat(parent);
#endif
        ss_pop(local_ss);
      }
      
      // release some nodes for stealing, if enough are available
      // and wake up quiescent threads
      releaseNodes(local_ss, remote_ss);
    }
		
    /* local work exhausted on this stack - resume tree search if able
     * to re-acquire work from shared portion of this thread's stack
     */
    if (ss_acquire(local_ss, chunkSize)) {
      continue;
    }

    // Do remote checks
    if (omp_get_thread_num() == 0) {
        if (ss_remote_acquire(remote_ss, local_ss, chunkSize)) {
            continue;
        }
    }

    /* no work left in this thread's stack           */
    /* try to steal work from another thread's stack */
    int goodSteal = 0;
    int victimId;

    const int n_local_attempts_before_remote = 2;
    ss_setState(local_ss, SS_SEARCH);
    do {
        int j;
        for (j = 0; j < n_local_attempts_before_remote && !goodSteal; j++) {
            victimId = findwork_local(chunkSize);
            if (victimId != -1) {
                goodSteal = ss_steal(local_ss, local_stealStack[victimId],
                        victimId, chunkSize, 1);
            }
        }

        if (!goodSteal && omp_get_thread_num() == 0) {
            victimId = findwork_remote(chunkSize);
            if (victimId != -1) {
                goodSteal = ss_steal(remote_ss, remote_stealStack[victimId],
                        victimId, chunkSize, 0);
            }
        }
    } while (victimId != -1 && !goodSteal);

    if (goodSteal) {
        continue;
    }
    // fprintf(stderr, "No good steal\n");
	
    /* unable to steal work from shared portion of other stacks -
     * enter quiescent state waiting for termination (done != 0)
     * or cancellation because some thread has made work available
     * (done == 0).
     */
    ss_setState(local_ss, SS_IDLE);
    done = cbarrier_wait(local_ss);
  }
  
  /* tree search complete ! */
}

// display search statistics
void showStats(double elapsedSecs) {
  int i;
  int tnodes = 0, tleaves = 0, trel = 0, tacq = 0, tsteal = 0, tfail= 0;
  int mdepth = 0, mheight = 0;
  double twork = 0.0, tsearch = 0.0, tidle = 0.0, tovh = 0.0, tcbovh = 0.0;

  /* Assemble all of the stealstacks so we can gather some stats. */
  for (i = 1; i < shmem_n_pes(); i++) {
      /* Collect up all of the StealStacks */
      shmem_getmem(remote_stealStack[i], remote_stealStack[i],
              sizeof(StealStack), i);
  }

  // combine measurements from all threads
  for (i = 0; i < shmem_n_pes(); i++) {
    tnodes  += remote_stealStack[i]->nNodes;
    tleaves += remote_stealStack[i]->nLeaves;
    trel    += remote_stealStack[i]->nRelease;
    tacq    += remote_stealStack[i]->nAcquire;
    tsteal  += remote_stealStack[i]->nSteal;
    tfail   += remote_stealStack[i]->nFail;
    twork   += remote_stealStack[i]->time[SS_WORK];
    tsearch += remote_stealStack[i]->time[SS_SEARCH];
    tidle   += remote_stealStack[i]->time[SS_IDLE];
    tovh    += remote_stealStack[i]->time[SS_OVH];
    tcbovh  += remote_stealStack[i]->time[SS_CBOVH];
    mdepth   = max(mdepth, remote_stealStack[i]->maxStackDepth);
    mheight  = max(mheight, remote_stealStack[i]->maxTreeDepth);
  }
  if (trel != tacq + tsteal) {
    printf("*** error! total released != total acquired + total stolen\n");
    printf("total released = %d, acquired = %d, stolen = %d\n", trel, tacq, tsteal);
  }
    
  uts_showStats(shmem_n_pes() * omp_get_num_threads(), chunkSize, elapsedSecs,
          tnodes, tleaves, mheight);

  // if (verbose > 1) {
  //   printf("Total chunks released = %d, of which %d reacquired and %d stolen\n",
  //       trel, tacq, tsteal);
  //   printf("Failed steal operations = %d, ", tfail);

  //   printf("Max stealStack size = %d\n", mdepth);
  //   printf("Avg time per thread: Work = %.6f, Search = %.6f, Idle = %.6f\n", (twork / GET_NUM_THREADS),
  //       (tsearch / GET_NUM_THREADS), (tidle / GET_NUM_THREADS));
  //   printf("                     Overhead = %6f, CB_Overhead = %6f\n\n", (tovh / GET_NUM_THREADS),
  //       (tcbovh/GET_NUM_THREADS));
  // }

  // per thread execution info
  // if (verbose > 2) {
  //   for (i = 0; i < GET_NUM_THREADS; i++) {
  //     printf("** Thread %d\n", i);
  //     printf("  # nodes explored    = %d\n", stealStack[i]->nNodes);
  //     printf("  # chunks released   = %d\n", stealStack[i]->nRelease);
  //     printf("  # chunks reacquired = %d\n", stealStack[i]->nAcquire);
  //     printf("  # chunks stolen     = %d\n", stealStack[i]->nSteal);
  //     printf("  # failed steals     = %d\n", stealStack[i]->nFail);
  //     printf("  maximum stack depth = %d\n", stealStack[i]->maxStackDepth);
  //     printf("  work time           = %.6f secs (%d sessions)\n",
  //            stealStack[i]->time[SS_WORK], stealStack[i]->entries[SS_WORK]);
  //     printf("  overhead time       = %.6f secs (%d sessions)\n",
  //            stealStack[i]->time[SS_OVH], stealStack[i]->entries[SS_OVH]);
  //     printf("  search time         = %.6f secs (%d sessions)\n",
  //            stealStack[i]->time[SS_SEARCH], stealStack[i]->entries[SS_SEARCH]);
  //     printf("  idle time           = %.6f secs (%d sessions)\n",
  //            stealStack[i]->time[SS_IDLE], stealStack[i]->entries[SS_IDLE]);
  //     printf("  wakeups             = %d, false wakeups = %d (%.2f%%)",
  //            stealStack[i]->wakeups, stealStack[i]->falseWakeups,
  //            (stealStack[i]->wakeups == 0) ? 0.00 : ((((double)stealStack[i]->falseWakeups)/stealStack[i]->wakeups)*100.0));
  //     printf("\n");
  //   }
  // }

  // tree statistics output to stat.txt, if requested
#ifdef UTS_STAT
  if (stats) {
    FILE *fp;
    char * tmpstr;
    char strBuf[5000];
    int  ind = 0;
    
    fp = fopen("stat.txt", "a+w");
    fprintf(fp, "\n------------------------------------------------------------------------------------------------------\n");
    ind = uts_paramsToStr(strBuf, ind);
    ind = impl_paramsToStr(strBuf, ind);
    //showParametersStr(strBuf);
    fprintf(fp, "%s\n", strBuf);
    
    fprintf(fp, "\nTotal nodes = %d\n", totalNodes); 
    fprintf(fp, "Max depth   = %d\n\n", maxHeight); 
    fprintf(fp, "Tseng ImbMeasure(overall)\n max:\t\t%lf \n avg:\t\t%lf \n devMaxAvg:\t %lf\n normDevMaxAvg: %lf\t\t\n\n", 
            imb_max/totalNodes, imb_avg/totalNodes, imb_devmaxavg/totalNodes, 
            imb_normdevmaxavg/totalNodes);
    
    switch (unbType){
    case 0: tmpstr = "(min imb weighted by size)"; break;
    case 1: tmpstr = "(min imb not weighted by size)"; break;
    case 2: tmpstr = "(max imb not weighted by size)"; break;
    default: tmpstr = "(?unknown measure)"; break;
    }
    fprintf(fp, "ImbMeasure:\t%s\n Overall:\t %lf\n Max:\t\t%lf\n Min:\t\t%lf\n\n", 
            tmpstr, treeImb, minImb, maxImb);
    showHist(fp);
    fprintf(fp, "\n------------------------------------------------------------------------------------------------------\n\n\n");
    fclose(fp);
  }
#endif
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

  shmem_init();
  // start_pes(0);

  /* determine benchmark parameters (all PEs) */
  uts_parseParams(argc, argv);

#ifdef UTS_STAT
  if (stats)
    initHist();
#endif  

  /* cancellable barrier initialization (single threaded under OMP) */
  cb_init();

#pragma omp parallel
#pragma omp single
  num_omp_threads = omp_get_num_threads();

  assert(shmem_n_pes() < MAX_SHMEM_THREADS);
  assert(num_omp_threads < MAX_OMP_THREADS);

  if (shmem_my_pe() == 0) {
      uts_printParams();
  }

  StealStack * ss_setup; 
  remote_stealStack[0] = (StealStack *)shmem_malloc(sizeof(StealStack));
  memset(remote_stealStack[0], 0, sizeof(StealStack));
  ss_setup = (StealStack *)remote_stealStack[0];
  ss_setup->stack_g = (Node *)shmem_malloc(MAXSTACKDEPTH * sizeof(Node));
  ss_setup->stack = (Node *)ss_setup->stack_g;
  ss_setup->remote_stackLock = smem_global_lock_alloc();
  ss_setup->local_stackLock = NULL;
  ss_init(ss_setup, MAXSTACKDEPTH);

  int i, j;
  for (i = 1; i < shmem_n_pes(); i++) {
      remote_stealStack[i] = (StealStack *)shmem_malloc(sizeof(StealStack));
      ss_setup = (StealStack *)remote_stealStack[i];
      ss_setup->stack_g = ss_setup->stack = remote_stealStack[0]->stack;
      ss_setup->remote_stackLock = smem_global_lock_alloc();
      ss_setup->local_stackLock = NULL;
      ss_init(ss_setup, MAXSTACKDEPTH);
  }

  for (i = 0; i < num_omp_threads; i++) {
      local_stealStack[i] = (StealStack *)malloc(sizeof(StealStack));
      ss_setup = (StealStack *)local_stealStack[i];
      ss_setup->stack_g = ss_setup->stack = (Node *)malloc(MAXSTACKDEPTH * sizeof(Node));
      ss_setup->remote_stackLock = NULL;
      ss_setup->local_stackLock = omp_global_lock_alloc();
      ss_init(ss_setup, MAXSTACKDEPTH);
  }

  /* initialize root node and push on thread 0 stack */
  if (shmem_my_pe() == 0) {
    initRootNode(&root, type);
    ss_push(local_stealStack[0], &root);
  }

  double t1;
  shmem_barrier_all();

/********** SPMD Parallel Region **********/
#pragma omp parallel shared(t1)
  {
    StealStack * local_ss = local_stealStack[omp_get_thread_num()];
    StealStack * remote_ss = (omp_get_thread_num() == 0 ?
            remote_stealStack[shmem_my_pe()] : NULL);

    /* time parallel search */
    ss_initState(local_ss);
    t1 = uts_wctime();
    parTreeSearch(local_ss, remote_ss);
  }

  shmem_barrier_all();
  const double t2 = uts_wctime();
  const double et = t2 - t1;

  // Collect stats on each OMP thread
  remote_stealStack[shmem_my_pe()]->nNodes = 0;
  remote_stealStack[shmem_my_pe()]->nLeaves = 0;
  remote_stealStack[shmem_my_pe()]->nRelease = 0;
  remote_stealStack[shmem_my_pe()]->nAcquire = 0;
  remote_stealStack[shmem_my_pe()]->nSteal = 0;
  remote_stealStack[shmem_my_pe()]->nFail = 0;
  remote_stealStack[shmem_my_pe()]->time[SS_WORK] = 0;
  remote_stealStack[shmem_my_pe()]->time[SS_SEARCH] = 0;
  remote_stealStack[shmem_my_pe()]->time[SS_IDLE] = 0;
  remote_stealStack[shmem_my_pe()]->time[SS_OVH] = 0;
  remote_stealStack[shmem_my_pe()]->time[SS_CBOVH] = 0;
  remote_stealStack[shmem_my_pe()]->maxStackDepth = 0;
  remote_stealStack[shmem_my_pe()]->maxTreeDepth = 0;
  for (j = 0; j < shmem_n_pes(); j++) {
      if (j == shmem_my_pe()) {
          for (i = 0; i < num_omp_threads; i++) {
              if (i == 0) {
                  fprintf(stderr, "PE %d thread %d nNodes=%d, # successful "
                          "remote acquire=%d, # failed remote acquire=%d, # "
                          "successful remote steals=%d, # failed remote "
                          "steals=%d, time waiting=%f s, # waits=%d, time "
                          "shared spinning=%f s, # shared spins=%d\n",
                          shmem_my_pe(), i,
                          local_stealStack[i]->nNodes,
                          local_stealStack[i]->nSuccessfulRemoteAcquire,
                          local_stealStack[i]->nFailedRemoteAcquire,
                          remote_stealStack[j]->nSuccessfulRemoteSteal,
                          remote_stealStack[j]->nFailedRemoteSteal,
                          local_stealStack[i]->timeInWait,
                          local_stealStack[i]->nWaits,
                          local_stealStack[i]->timeInSharedSpin,
                          local_stealStack[i]->nSharedSpins);
              } else {
                  fprintf(stderr, "PE %d thread %d nNodes=%d, time waiting=%f "
                          "s, # waits=%d\n", shmem_my_pe(), i, local_stealStack[i]->nNodes,
                          local_stealStack[i]->timeInWait,
                          local_stealStack[i]->nWaits);
              }
              remote_stealStack[shmem_my_pe()]->nNodes += local_stealStack[i]->nNodes;
              remote_stealStack[shmem_my_pe()]->nLeaves += local_stealStack[i]->nLeaves;
              remote_stealStack[shmem_my_pe()]->nRelease += local_stealStack[i]->nRelease;
              remote_stealStack[shmem_my_pe()]->nAcquire += local_stealStack[i]->nAcquire;
              remote_stealStack[shmem_my_pe()]->nSteal += local_stealStack[i]->nSteal;
              remote_stealStack[shmem_my_pe()]->nFail += local_stealStack[i]->nFail;
              remote_stealStack[shmem_my_pe()]->time[SS_WORK] += local_stealStack[i]->time[SS_WORK];
              remote_stealStack[shmem_my_pe()]->time[SS_SEARCH] += local_stealStack[i]->time[SS_SEARCH];
              remote_stealStack[shmem_my_pe()]->time[SS_IDLE] += local_stealStack[i]->time[SS_IDLE];
              remote_stealStack[shmem_my_pe()]->time[SS_OVH] += local_stealStack[i]->time[SS_OVH];
              remote_stealStack[shmem_my_pe()]->time[SS_CBOVH] += local_stealStack[i]->time[SS_CBOVH];
              remote_stealStack[shmem_my_pe()]->maxStackDepth = max(
                      local_stealStack[i]->maxStackDepth,
                      remote_stealStack[shmem_my_pe()]->maxStackDepth);
              remote_stealStack[shmem_my_pe()]->maxTreeDepth = max(
                      local_stealStack[i]->maxTreeDepth,
                      remote_stealStack[shmem_my_pe()]->maxTreeDepth);
          }
          shmem_barrier_all();
      }
  }
  shmem_barrier_all();

  /* display results */
  if (shmem_my_pe() == 0) {
    showStats(et);
  }

/********** End Parallel Region **********/

  shmem_finalize();

  return 0;
}
