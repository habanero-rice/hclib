/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hclib-async-struct.h"
#include "hclib.h"

#ifndef HCUPC_SUPPORT_H_
#define HCUPC_SUPPORT_H_

#if defined(HUPCPP) && defined(DIST_WS)

typedef struct asyncAnyInfo {
	int asyncAny_pushed;
	volatile int asyncAny_stolen;
} asyncAnyInfo;

#define MAX_REMOTE_ASYNCANY_ARG_SIZE 384

struct remoteAsyncAny_task  {
	generic_frame_ptr _fp;
	char _args[MAX_REMOTE_ASYNCANY_ARG_SIZE];
	inline void init_remoteAsyncAny_task(generic_frame_ptr fp, size_t arg_sz, void *remoteAsyncAny_args) {
		HASSERT(arg_sz <= MAX_REMOTE_ASYNCANY_ARG_SIZE);
		this->_fp = fp;
		if (arg_sz > 0) {
			memcpy(&this->_args, remoteAsyncAny_args, arg_sz);
		}
	}

	inline remoteAsyncAny_task() { };

	template<typename Function, typename T1>
	inline remoteAsyncAny_task(Function kernel, const T1 &a1) {
		async_arguments1<Function, T1> args(kernel, a1);
		init_remoteAsyncAny_task(wrapper1<Function, T1>, (size_t)sizeof(args), (void *)&args);
	}
};

typedef struct commWorkerAsyncAny_infoStruct_t {
	remoteAsyncAny_task* ptr_to_outgoingAsyncAny;
	bool initiatePackingOfAsyncAny;
} commWorkerAsyncAny_infoStruct_t;

extern commWorkerAsyncAny_infoStruct_t *commWorkerAsyncAny_infoStruct;

template <typename T>
inline void execute_remoteAsyncAny_task(T lambda) {
	lambda();
}

template <typename T>
inline void execute_hcupc_lambda(T* lambda) {
	const int wid = get_current_worker();
	if(wid != 0) {
		// only computation workers can enter
		MARK_BUSY(wid);
		(*lambda)();
		MARK_OVH(wid);
	}
	else {
		// only communication worker can enter
		if(!commWorkerAsyncAny_infoStruct->initiatePackingOfAsyncAny) {	// fast path
			(*lambda)();
		}
		else {	//slow path
			// only computation workers can enter to pack lambda for remote asyncAny
			/*
			 * UPC++ async can only ship a stack allocated lambda to remote place.
			 * If a lambda is heap allocated then the pointer will not be valid at
			 * remote place. Due to this we have to first copy the content of this
			 * lambda into a stack allocated lambda.
			 *
			 * XXX - This description doesn't make any sense to me.
			 * There's no way in C++ to differentiate a pointer to
			 * an object with dynamic vs automatic storage duration,
			 * so I don't see how this could be true. To further
			 * complicate this, we're currently using heap-allocated
			 * stacks for our fibers, so everything is in the heap
			 * (i.e., even address-range-checking wouldn't work for
			 * differentiating between stack and heap). This will
			 * need to be refactored later, but I need to find out
			 * what exactly it's supposed to be doing first.
			 */

			/*
			 * 1. Copy into a char array.
			 * This user lambda does not have a default constructor / copy constructor and hence we cannot
			 * create a new lambda directly out of this lambda. Hence here we copy this into a char array
			 */
			char array[sizeof(T)];
			memcpy((void*)array, lambda, sizeof(T));
			// 2. char array is stack allocated and is ready to ship to remote place when packed into another stack allocated labmda
			auto remote_lambda = [array]() {
				(*((T*)array))();
			};
			// 3. copy this remote lambda into a struct of type which is also known to the remote place
			remoteAsyncAny_task tmp = remoteAsyncAny_task(execute_remoteAsyncAny_task<decltype(remote_lambda)>, remote_lambda);
			memcpy(commWorkerAsyncAny_infoStruct->ptr_to_outgoingAsyncAny, &tmp, sizeof(remoteAsyncAny_task));
		}
	}
	// FIXME - this whole call chain is kind of a mess
	// leaving C malloc/free and memcpy calls for now (come back to fix it later)
	free((void*)lambda);
}

template <typename T>
inline hclib_task_t* _allocate_async_hcupc(T lambda, bool await) {
	// FIXME - this whole call chain is kind of a mess
	// leaving C malloc/free and memcpy calls for now (come back to fix it later)
	const size_t hclib_task_size = await ? sizeof(hclib_dependent_task_t) : sizeof(hclib_task_t);
	hclib_task_t* task = (hclib_task_t*) malloc(hclib_task_size);
	const size_t lambda_size = sizeof(T);
	T* lambda_onHeap = (T*) malloc(lambda_size);
	memcpy(lambda_onHeap, &lambda, lambda_size);
	hclib_task_t t = hclib_task_t(execute_hcupc_lambda<T>, lambda_onHeap);
	memcpy(task, &t, sizeof(hclib_task_t));
	return task;
}

void spawn_asyncAnyTask(hclib_task_t * task);

template <typename T>
inline void asyncAny(T lambda) {
	MARK_OVH(current_ws()->id);
	hclib_task_t* task = _allocate_async_hcupc<T>(lambda, false);
	spawn_asyncAnyTask(task);
}
#endif	/* defined(HUPCPP) && defined(DIST_WS) */

/*
 * some functions which are independent of hcupc implementation
 */
void init_hcupc_related_datastructures(int w);
void free_hcupc_related_datastructures();
void check_if_hcupc_distributed_futures(hclib_future_t** future_list);
void hcupc_reset_asyncAnyInfo(int id);
void hcupc_check_if_asyncAny_stolen(hclib_task_t* buff, int victim, int id);
void hcupc_inform_failedSteal(int id);
void hcupc_check_if_asyncAny_pop(hclib_task_t* buff, int id);

#ifdef HUPCPP
#ifdef __cplusplus
extern "C" {
#endif
void hclib_gather_comm_worker_stats(int* push_outd, int* push_ind,
        int* steal_ind);
int totalPendingLocalAsyncs();
void hclib_display_runtime();
volatile int* hclib_start_finish_special();
#ifdef __cplusplus
}
#endif
#endif

#ifdef DIST_WS
int totalAsyncAnyAvailable();
bool steal_fromComputeWorkers_forDistWS(remoteAsyncAny_task* remAsyncAnybuff);
void registerHCUPC_callback(volatile int*);
#endif

#endif /* HCUPC_SUPPORT_H_ */
