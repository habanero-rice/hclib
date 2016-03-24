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
 * hcupc-support.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
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
	HC_FREE((void*)lambda);
}

template <typename T>
inline hclib_task_t* _allocate_async_hcupc(T lambda, bool await) {
	const size_t hclib_task_size = await ? sizeof(hclib_dependent_task_t) : sizeof(hclib_task_t);
	hclib_task_t* task = (hclib_task_t*) HC_MALLOC(hclib_task_size);
	const size_t lambda_size = sizeof(T);
	T* lambda_onHeap = (T*) HC_MALLOC(lambda_size);
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
