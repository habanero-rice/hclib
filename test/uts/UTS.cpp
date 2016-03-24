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

#include "hclib_cpp.h"
#include <string.h>
#include <float.h>

#include "uts.h"

/* Search status */
#define STATUS_HAVEWORK 0
#define STATUS_NOWORK   1
#define SS_NSTATES 4

/* data per thread */
struct stealStack_t {
	long localWork;     /* amount of local only work*/
	counter_t nNodes, nLeaves;  /* stats */
	Node *stack;
	int maxStackDepth;
	int stack_head, stack_tail;
	int maxTreeDepth;
	int root;
	int chunk_size, work_size;
};

typedef struct stealStack_t StealStack;

#define MAXSTACKDEPTH               1048576 //1M Nodes

// parallel execution parameters 
int chunkSize = 20;    // number of nodes to move to/from shared area
int polling_interval = 0;
static size_t work_chunk_size;

/** Global Parameters: Set in ss_init() **/
static int nthreads;

/** Adaptive Polling Interval Parameters **/
static int pollint_default    = 0; // 0 is adaptive
/*
static int pollint_isadaptive = 0;
static int pollint_min        = 1;
static int pollint_max        = 1024;

#define POLLINT_GROW   polling_interval + 4
#define POLLINT_SHRINK pollint_min / 2
//#define POLLINT_SHRINK pollint_min
 */

/** Global arrays of worker specific states **/
static StealStack    **threadStealStacks; // worker specific UTS-related stats

// forward declarations

void ss_init(int *argc, char ***argv);
int  ss_start(size_t work_size, int chunk_size);
void ss_finalize();
void ss_error(char *str, int error);
void ss_initialize(StealStack * s);
int  ss_get_work(int wid, void* node_c);
void showStats(double walltime);
static void parTreeSearch();


/***********************************************************
 *  UTS Implementation                                     *
 ***********************************************************/

using namespace std;

int main(int argc, char *argv[]) {
	hclib::launch(&argc, argv, [&]() {

        double t1, t2;
        /* initialize stealstacks and comm. layer */
        ss_init(&argc, &argv);

        /* determine benchmark parameters */
        uts_parseParams(argc, argv);

        /* show parameter settings */
        uts_printParams();

        ss_start(sizeof(Node), chunkSize);

        /* time parallel search */
        t1 = uts_wctime();

        hclib::finish([=] {
            hclib::async([=]() {
                int wid = hclib::current_worker();
                /* initialize root node and push on thread 0 stack */
                StealStack *s = threadStealStacks[wid];
                s->root = 1;
                Node *root = &(s->stack[s->stack_head]);
                uts_initRoot(root, type);
                s->stack_head++;
                s->localWork++;
                s->maxStackDepth = max(s->localWork, s->maxStackDepth);

                parTreeSearch();
            });
        });

        t2 = uts_wctime();

        /* display results */
        showStats(t2 - t1);

        ss_finalize();

    });

	return 0;
}

/* Fatal error */
void ss_error(char *str, int error)
{
	fprintf(stderr, "*** Error: %s\n", str);
	exit(1);
}

void push_surplusNodes(const Node* work) {
	int wid = hclib::current_worker();
	StealStack *ss = threadStealStacks[wid];
	ss->localWork = chunkSize;
	ss->stack_head = chunkSize;
	ss->stack_tail = 0;
	memcpy(&(ss->stack[0]), work, work_chunk_size);

	parTreeSearch();
}

/* 
 * Generate all children of the parent
 *
 * details depend on tree type, node type and shape function
 *
 */
void genChildren(Node * parent, int wid) {
	int parentHeight = parent->height;
	int numChildren, childType;
	StealStack *ss = threadStealStacks[wid];

	ss->maxTreeDepth = max(ss->maxTreeDepth, parent->height);

	numChildren = uts_numChildren(parent);
	childType   = uts_childType(parent);

	// record number of children in parent
	parent->numChildren = numChildren;

	// construct children and push onto stack
	if (numChildren > 0) {
		int i, j;
		int localSize = 2*chunkSize;
		int stack_size = numChildren + ss->stack_head - ss->stack_tail;
		if (stack_size >= MAXSTACKDEPTH) {
			printf("Stack Head: %d Tail: %d\n", ss->stack_head, ss->stack_tail);
			ss_error("Worker stack out of memory", 10);
		}

		for (i = 0; i < numChildren; i++) {
			Node * child = &(ss->stack[ss->stack_head]);
			ss->stack_head++;
			child->type = childType;
			child->height = parentHeight + 1;

			for (j = 0; j < computeGranularity; j++) {
				// TBD:  add parent height to spawn
				// computeGranularity controls number of rng_spawn calls per node
				rng_spawn(parent->state.state, child->state.state, i);
			}

			ss->localWork++;
			ss->maxStackDepth = max(ss->localWork, ss->maxStackDepth);

			/* If there is sufficient local work, release a chunk to the global queue */
			if (ss->localWork > localSize) {
				if ((ss->nNodes % polling_interval) == 0) {
					assert(chunkSize <= 20);
					Node work[20];	//TODO: C++11 lambda currently does not allow capturing variable length arrays
					//void * work = malloc(work_chunk_size);
					memcpy(work, &(ss->stack[ss->stack_tail]), work_chunk_size);
					ss->stack_tail += chunkSize;
					ss->localWork -= chunkSize;
					hclib::async([work]() {
						push_surplusNodes(work);
					});
				}
			}
		}
	} else {
		ss->nLeaves++;
	}
}

/* 
 * parallel search of UTS trees using work stealing 
 * 
 *   Note: tree size is measured by the number of
 *         push operations
 */

static void parTreeSearch() {
	int wid = hclib::current_worker();
	StealStack *ss = threadStealStacks[wid];
	int root = ss->root;
	Node parent;
	int status = ss_get_work(wid, &parent);

	if (status == STATUS_HAVEWORK) {
		do {
			genChildren(&parent, wid);
			status = ss_get_work(wid, &parent);
		} while (status == STATUS_HAVEWORK);
	}
}

void showStats(double walltime) {
	int i;
	counter_t tnodes = 0, tleaves = 0, trel=0, tacq=0;
	counter_t mdepth = 0, mheight = 0;
	double elapsedSecs;

	elapsedSecs = walltime;

	// combine measurements from all threads
	for (int i = 0; i < nthreads; i++) {
		StealStack *ss = threadStealStacks[i];
		tnodes  += ss->nNodes;
		tleaves += ss->nLeaves;
		mdepth   = max(mdepth, ss->maxStackDepth);
		mheight  = max(mheight, ss->maxTreeDepth);
	}

	uts_showStats(1, chunkSize, elapsedSecs, tnodes, tleaves, mheight);
}


/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// Return a string describing this implementation
char * impl_getName() {
	return "HabaneroUPC++ work-stealing UTS";
}


// construct string with all parameter settings
int impl_paramsToStr(char *strBuf, int ind) {
	ind += sprintf(strBuf+ind, "Execution strategy:  ");
	ind += sprintf(strBuf+ind, "Parallel search using %d threads\n", nthreads);
	ind += sprintf(strBuf+ind, "   Load balance by work stealing, chunk size = %d nodes\n",chunkSize);
	ind += sprintf(strBuf+ind, "   Polling Interval: %d\n", polling_interval);

	return ind;
}

// Parse command-line flags
int impl_parseParam(char *param, char *value) {
	int err = 0;  // Return 0 on a match, nonzero on an error

	switch (param[1]) {
	case 'c':
		chunkSize = atoi(value); break;
	case 'i':
		polling_interval = atoi(value); break;
	default:
		err = 1;
		break;
	}

	return err;
}

// Add this to the generic help message
void impl_helpMessage() {
	printf("   -c  int   chunksize for work sharing and work stealing\n");
	printf("   -i  int   work stealing/sharing interval (stealing default: adaptive)\n");
}


void impl_abort(int err) {
	exit(1);
}

/***********************************************************
 *  CRT Workstealing Implementation                        *
 ***********************************************************/

/* Fatal error */
int ss_get_num_processes() {
	return 1;
}

char * ss_get_par_description()
{
	return "CRT UTS-2";
}

/* Num threads per process */
int ss_get_num_threads()
{
	return nthreads;
}

/* initialize the stack */
void ss_init(int *argc, char ***argv)
{
	int i;
	nthreads = hclib::num_workers();

	/* Worker specific init */
	threadStealStacks = (StealStack **)malloc(sizeof(StealStack**) * nthreads);
	for (i = 0; i < nthreads; i++) {
		threadStealStacks[i] = (StealStack *)malloc(sizeof(StealStack));
		ss_initialize(threadStealStacks[i]);
		threadStealStacks[i]->stack = (Node*)malloc(sizeof(Node) * MAXSTACKDEPTH);
	}

	// Set a default polling interval
	polling_interval = pollint_default;
}

void ss_initialize(StealStack * s)
{
	s->localWork     = 0;
	s->nNodes        = 0;
	s->nLeaves       = 0;
	s->maxStackDepth = 0;
	s->maxTreeDepth  = 0;
	s->root          = 0;
	s->stack_head    = 0;
	s->stack_tail    = 0;
}

int ss_start(size_t work_size, int chunk_size)
{
	int i, j;

	work_chunk_size = chunk_size*work_size;

	// Using adaptive polling interval?
	if (polling_interval == 0) {
		/*pollint_isadaptive = 1;*/
		polling_interval   = 1;
	}

	return 1;
}

void ss_finalize()
{
	for (int i = 0; i < nthreads; i++) {
		delete(threadStealStacks[i]->stack);
		delete(threadStealStacks[i]);
	}
	delete(threadStealStacks);
}

/**
   if no work is found no local work is found, and
   none can be stolen, return original s and c is null
   if work is found, return the StealStack and set c to 
   return node
 **/
int ss_get_work(int wid, void* node_c)
{
	StealStack *s = threadStealStacks[wid];

	if (s->localWork != 0) {

		/* ensureLocalWork() ensures that the local work queue is not empty,
		 * so at this point we know there must be work available */

		/* head always points at the next free entry in the work array */
		s->stack_head--;
		memcpy(node_c, &(s->stack[s->stack_head]), sizeof(Node));

		s->localWork--;
		s->nNodes++;
		return STATUS_HAVEWORK;
	}

	return STATUS_NOWORK;
}
