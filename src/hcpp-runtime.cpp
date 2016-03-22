/* Copyright (c) 2015, Rice University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
3.  Neither the name of Rice University
     nor the names of its contributors may be used to endorse or
     promote products derived from this software without specific
     prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

/*
 * hcpp-runtime.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include "hcpp-internal.h"
#include <pthread.h>
#include "hcpp-atomics.h"
#include <sys/time.h>

namespace hcpp {

using namespace std;

static double benchmark_start_time_stats = 0;
static double user_specified_timer = 0;
pthread_key_t wskey;
pthread_once_t selfKeyInitialized = PTHREAD_ONCE_INIT;

#ifdef HCPP_COMM_WORKER
semiConcDeque_t * comm_worker_out_deque;
#endif

static finish_t*	root_finish;

hc_context* 		hcpp_context;
hc_options* 		hcpp_options;

void log_(const char * file, int line, hc_workerState * ws, const char * format, ...) {
	va_list l;
	FILE * f = stderr;
	if (ws != NULL) {
		fprintf(f, "[worker: %d (%s:%d)] ", ws->id, file, line);
	} else {
		fprintf(f, "[%s:%d] ", file, line);
	}
	va_start(l, format);
	vfprintf(f, format, l);
	fflush(f);
	va_end(l);
}

// Statistics
int total_push_outd;
int* total_push_ind;
int* total_steals;

inline void increment_async_counter(int wid) {
	total_push_ind[wid]++;
}

inline void increment_steals_counter(int wid) {
	total_steals[wid]++;
}

inline void increment_asyncComm_counter() {
	total_push_outd++;
}

// One global finish scope

static void initializeKey() {
	pthread_key_create(&wskey, NULL);
}

void set_current_worker(int wid) {
	pthread_setspecific(wskey, hcpp_context->workers[wid]);
	if(getenv("HCPP_BIND_THREADS")) {
		bind_thread(wid, NULL, 0);
	}
}

int get_current_worker() {
	return ((hc_workerState*)pthread_getspecific(wskey))->id;
}

hc_workerState* current_ws() {
	return current_ws_internal();
}

//FWD declaration for pthread_create
void * worker_routine(void * args);

void hcpp_global_init(bool HPT) {
	// Build queues
	hcpp_context->done = 1;
	if(!HPT) {
		hcpp_context->nproc = hcpp_options->nproc;
		hcpp_context->nworkers = hcpp_options->nworkers;
		hcpp_context->options = hcpp_options;

		place_t * root = (place_t*) malloc(sizeof(place_t));
		root->id = 0;
		root->nnext = NULL;
		root->child = NULL;
		root->parent = NULL;
		root->type = MEM_PLACE;
		root->ndeques = hcpp_context->nworkers;
		hcpp_context->hpt = root;
		hcpp_context->places = &hcpp_context->hpt;
		hcpp_context->nplaces = 1;
		hcpp_context->workers = (hc_workerState**) malloc(sizeof(hc_workerState*) * hcpp_options->nworkers);
		HASSERT(hcpp_context->workers);
		hc_workerState * cur_ws = NULL;
		for(int i=0; i<hcpp_options->nworkers; i++) {
			hcpp_context->workers[i] = new hc_workerState;
			HASSERT(hcpp_context->workers[i]);
			hcpp_context->workers[i]->context = hcpp_context;
			hcpp_context->workers[i]->id = i;
			hcpp_context->workers[i]->pl = root;
			hcpp_context->workers[i]->hpt_path = NULL;
			hcpp_context->workers[i]->nnext = NULL;
			if (i == 0) {
				cur_ws = hcpp_context->workers[i];
			} else {
				cur_ws->nnext = hcpp_context->workers[i];
				cur_ws = hcpp_context->workers[i];
			}
		}
		root->workers = hcpp_context->workers[0];
	}
	else {
		hcpp_context->hpt = readhpt(&hcpp_context->places, &hcpp_context->nplaces, &hcpp_context->nproc, &hcpp_context->workers, &hcpp_context->nworkers);
		for (int i=0; i<hcpp_context->nworkers; i++) {
			hc_workerState * ws = hcpp_context->workers[i];
			ws->context = hcpp_context;
		}
	}

	total_push_outd = 0;
	total_steals = new int[hcpp_context->nworkers];
	total_push_ind = new int[hcpp_context->nworkers];
	for(int i=0; i<hcpp_context->nworkers; i++) {
		total_steals[i] = 0;
		total_push_ind[i] = 0;
	}

#ifdef HCPP_COMM_WORKER
	comm_worker_out_deque = new semiConcDeque_t;
	HASSERT(comm_worker_out_deque);
	semiConcDequeInit(comm_worker_out_deque, NULL);
#endif

	init_hcupc_related_datastructures(hcpp_context->nworkers);
}

void hcpp_createWorkerThreads(int nb_workers) {
	/* setting current thread as worker 0 */
	// Launch the worker threads
	pthread_once(&selfKeyInitialized, initializeKey);
	// Start workers
	for(int i=1;i<nb_workers;i++) {
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		pthread_create(&hcpp_context->workers[i]->t, &attr, &worker_routine, &hcpp_context->workers[i]->id);
	}
	set_current_worker(0);
}

void display_runtime() {
	cout << "---------HCPP_RUNTIME_INFO-----------" << endl;
	printf(">>> HCPP_WORKERS\t\t= %s\n",getenv("HCPP_WORKERS"));
	printf(">>> HCPP_HPT_FILE\t= %s\n",getenv("HCPP_HPT_FILE"));
	printf(">>> HCPP_BIND_THREADS\t= %s\n",getenv("HCPP_BIND_THREADS"));
	if(getenv("HCPP_WORKERS") && getenv("HCPP_BIND_THREADS")) {
		printf("WARNING: HCPP_BIND_THREADS assign cores in round robin. E.g., setting HCPP_WORKERS=12 on 2-socket node, each with 12 cores, will assign both HCUPC++ places on same socket\n");
	}
	printf(">>> HCPP_STATS\t\t= %s\n",getenv("HCPP_STATS"));
	cout << "----------------------------------------" << endl;
}

void hcpp_entrypoint(bool HPT) {
	if(getenv("HCPP_STATS")) {
		display_runtime();
	}

	srand(0);

	hcpp_options = new hc_options;
	HASSERT(hcpp_options);
	hcpp_context = new hc_context;
	HASSERT(hcpp_context);

	if(!HPT) {
		char* workers_env = getenv("HCPP_WORKERS");
		int workers = 1;
		if(!workers_env) {
			std::cout << "HCPP: WARNING -- Number of workers not set. Please set using env HCPP_WORKERS" << std::endl;
		}
		else {
			workers = atoi(workers_env);
		}
		HASSERT(workers > 0);
		hcpp_options->nworkers = workers;
		hcpp_options->nproc = workers;
	}

	hcpp_global_init(HPT);

#ifdef __USE_HC_MM__
	const char* mm_alloc_batch_size = getenv("HCPP_MM_ALLOCBATCHSIZE");
	const int mm_alloc_batch_size_int = mm_alloc_batch_size ? atoi(mm_alloc_batch_size) : HC_MM_ALLOC_BATCH_SIZE;
	hcpp_options->alloc_batch_size = mm_alloc_batch_size_int;
	hc_mm_init(hcpp_context);
#endif

	hc_hpt_init(hcpp_context);
#if TODO
	hc_hpt_dev_init(hcpp_context);
#endif

	// init timer stats
#ifdef HCPP_COMM_WORKER
	hcpp_initStats(hcpp_context->nworkers, true);
#else
	hcpp_initStats(hcpp_context->nworkers, false);
#endif

	/* Create key to store per thread worker_state */
	if (pthread_key_create(&wskey, NULL) != 0) {
		log_die("Cannot create wskey for worker-specific data");
	}

	/* set pthread's concurrency. Doesn't seem to do much on Linux */
	pthread_setconcurrency(hcpp_context->nworkers);

	/* Create all worker threads  */
	hcpp_createWorkerThreads(hcpp_context->nworkers);

	// allocate root finish
	root_finish = new finish_t;
	current_ws_internal()->current_finish = root_finish;
	start_finish();
}

void hcpp_join_workers(int nb_workers) {
	// Join the workers
	hcpp_context->done = 0;
	for(int i=1;i< nb_workers; i++) {
		pthread_join(hcpp_context->workers[i]->t, NULL);
	}
}

void hcpp_cleanup() {
	hc_hpt_dev_cleanup(hcpp_context);
	hc_hpt_cleanup_1(hcpp_context); /* cleanup deques (allocated by hc mm) */
#ifdef USE_HC_MM
	hc_mm_cleanup(hcpp_context);
#endif
	hc_hpt_cleanup_2(hcpp_context); /* cleanup the HPT, places, and workers (allocated by malloc) */
	pthread_key_delete(wskey);

	free(hcpp_context);
	free(hcpp_options);
	free(total_steals);
	free(total_push_ind);
	free_hcupc_related_datastructures();
}

inline void check_in_finish(finish_t * finish) {
	hc_atomic_inc(&(finish->counter));
}

inline void check_out_finish(finish_t * finish) {
	hc_atomic_dec(&(finish->counter));
}

inline void execute_task(task_t* task) {
	finish_t* current_finish = task->get_current_finish();
	current_ws_internal()->current_finish = current_finish;

	(task->_fp)(task->_args);
	check_out_finish(current_finish);
	HC_FREE(task);
}

inline void rt_schedule_async(task_t* async_task, int comm_task) {
	if(comm_task) {
#ifdef HCPP_COMM_WORKER
		// push on comm_worker out_deq
		semiConcDequeLockedPush(comm_worker_out_deque, (task_t*) async_task);
#endif
	}
	else {
		// push on worker deq
		if(!dequePush(&(hcpp_context->workers[get_current_worker()]->current->deque), (task_t*) async_task)) {
			// TODO: deque is full, so execute in place
			printf("WARNING: deque full, local executino\n");
			execute_task((task_t*) async_task);
		}
	}
}

inline int is_eligible_to_schedule(task_t * async_task) {
    if (async_task->ddf_list != NULL) {
    	struct ddt_st * ddt = (ddt_t *) rt_async_task_to_ddt(async_task);
        return iterate_ddt_frontier(ddt);
    } else {
        return 1;
    }
}

void try_schedule_async(task_t * async_task, int comm_task) {
    if (is_eligible_to_schedule(async_task)) {
        rt_schedule_async(async_task, comm_task);
    }
}

void spawn_at_hpt(place_t* pl, task_t * task) {
	// get current worker
	hc_workerState* ws = current_ws_internal();
	check_in_finish(ws->current_finish);
	task->set_current_finish(ws->current_finish);
	deque_push_place(ws, pl, task);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif
}

void spawn(task_t * task) {
	// get current worker
	hc_workerState* ws = current_ws_internal();
	check_in_finish(ws->current_finish);
	task->set_current_finish(ws->current_finish);
	try_schedule_async(task, 0);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif

}

void spawn_await(task_t * task, ddf_t** ddf_list) {
	// check if this is DDDf_t (remote or owner) and do callback to HabaneroUPC++ for implementation
	check_if_hcupc_dddf(ddf_list);
	// get current worker
	hc_workerState* ws = current_ws_internal();
	check_in_finish(ws->current_finish);
	task->set_current_finish(ws->current_finish);
	task->set_ddf_list(ddf_list);
	hcpp_task_t *t = (hcpp_task_t*) task;
	ddt_init(&(t->ddt), ddf_list);
	try_schedule_async(task, 0);
#ifdef HC_COMM_WORKER_STATS
	const int wid = get_current_worker();
	increment_async_counter(wid);
#endif

}

void spawn_commTask(task_t * task) {
#ifdef HCPP_COMM_WORKER
	hc_workerState* ws = current_ws_internal();
	check_in_finish(ws->current_finish);
	task->set_current_finish(ws->current_finish);
	try_schedule_async(task, 1);
#else
	assert(false);
#endif
}

inline void slave_worker_finishHelper_routine(finish_t* finish) {
	hc_workerState* ws = current_ws_internal();
	int wid = ws->id;

	while(finish->counter > 0) {
		// try to pop
		task_t* task = hpt_pop_task(ws);
		if (!task) {
			while(finish->counter > 0) {
				// try to steal
				task = hpt_steal_task(ws);
				if (task) {
#ifdef HC_COMM_WORKER_STATS
					increment_steals_counter(wid);
#endif
					break;
				}
			}
		}
		if(task) {
			execute_task(task);
		}
	}
}

#ifdef HCPP_COMM_WORKER
inline void master_worker_routine(finish_t* finish) {
	semiConcDeque_t *deque = comm_worker_out_deque;
	while(finish->counter > 0) {
		// try to pop
		task_t* task = semiConcDequeNonLockedPop(deque);
		// Comm worker cannot steal
		if(task) {
#ifdef HC_COMM_WORKER_STATS
			increment_asyncComm_counter();
#endif
			execute_task(task);
		}
	}
}
#endif

void* worker_routine(void * args) {
	int wid = *((int *) args);
	set_current_worker(wid);

	hc_workerState* ws = current_ws_internal();

	while(hcpp_context->done) {
		task_t* task = hpt_pop_task(ws);
		if (!task) {
			while(hcpp_context->done) {
				// try to steal
				task = hpt_steal_task(ws);
				if (task) {
#ifdef HC_COMM_WORKER_STATS
					increment_steals_counter(wid);
#endif
					break;
				}
			}
		}

		if(task) {
			execute_task(task);
		}
	}

	return NULL;
}

void teardown() {

}

inline void help_finish(finish_t * finish) {
#ifdef HCPP_COMM_WORKER
	if(current_ws_internal()->id == 0) {
		master_worker_routine(finish);
	}
	else {
		slave_worker_finishHelper_routine(finish);
	}
#else
	slave_worker_finishHelper_routine(finish);
#endif
}

/*
 * =================== INTERFACE TO USER FUNCTIONS ==========================
 */

void start_finish() {
	hc_workerState* ws = current_ws_internal();
	finish_t * finish = (finish_t*) HC_MALLOC(sizeof(finish_t));
	finish->counter = 0;
	finish->parent = ws->current_finish;
	if(finish->parent) {
		check_in_finish(finish->parent);
	}
	ws->current_finish = finish;
}

void end_finish() {
	hc_workerState* ws =current_ws_internal();
	finish_t* current_finish = ws->current_finish;

	if (current_finish->counter > 0) {
		help_finish(current_finish);
	}
	HASSERT(current_finish->counter == 0);

	if(current_finish->parent) {
		check_out_finish(current_finish->parent);
	}

	ws->current_finish = current_finish->parent;
	HC_FREE(current_finish);
}

void finish(std::function<void()> lambda) {
	start_finish();
	lambda();
	end_finish();
}

int numWorkers() {
	return hcpp_context->nworkers;
}

int get_hc_wid() {
	return get_current_worker();
}

void gather_commWorker_Stats(int* push_outd, int* push_ind, int* steal_ind) {
	int asyncPush=0, steals=0, asyncCommPush=total_push_outd;
	for(int i=0; i<numWorkers(); i++) {
		asyncPush += total_push_ind[i];
		steals += total_steals[i];
	}
	*push_outd = asyncCommPush;
	*push_ind = asyncPush;
	*steal_ind = steals;
}

double mysecond() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + ((double) tv.tv_usec / 1000000);
}

void runtime_statistics(double duration) {
	int asyncPush=0, steals=0, asyncCommPush=total_push_outd;
	for(int i=0; i<numWorkers(); i++) {
		asyncPush += total_push_ind[i];
		steals += total_steals[i];
	}

	double tWork, tOvh, tSearch;
	hcpp_getAvgTime (&tWork, &tOvh, &tSearch);

	double total_duration = user_specified_timer>0 ? user_specified_timer : duration;
	printf("============================ MMTk Statistics Totals ============================\n");
	printf("time.mu\ttotalPushOutDeq\ttotalPushInDeq\ttotalStealsInDeq\ttWork\ttOverhead\ttSearch\n");
	printf("%.3f\t%d\t%d\t%d\t%.4f\t%.4f\t%.5f\n",total_duration,asyncCommPush,asyncPush,steals,tWork,tOvh,tSearch);
	printf("Total time: %.3f ms\n",total_duration);
	printf("------------------------------ End MMTk Statistics -----------------------------\n");
	printf("===== TEST PASSED in %.3f msec =====\n",duration);
}

void showStatsHeader() {
	cout << endl;
	cout << "-----" << endl;
	cout << "mkdir timedrun fake" << endl;
	cout << endl;
	cout << "-----" << endl;
	benchmark_start_time_stats = mysecond();
}

void user_harness_timer(double dur) {
	user_specified_timer = dur;
}

void showStatsFooter() {
	double end = mysecond();
	HASSERT(benchmark_start_time_stats != 0);
	double dur = (end-benchmark_start_time_stats)*1000;
	runtime_statistics(dur);
}

void init(int * argc, char ** argv) {
	if(getenv("HCPP_STATS")) {
		showStatsHeader();
	}
	// get the total number of workers from the env
	const bool HPT = getenv("HCPP_HPT_FILE") != NULL;
	hcpp_entrypoint(HPT);
}

void finalize() {
	end_finish();
	free(root_finish);

	if(getenv("HCPP_STATS")) {
		showStatsFooter();
	}

	hcpp_join_workers(hcpp_context->nworkers);
	hcpp_cleanup();
}

}
