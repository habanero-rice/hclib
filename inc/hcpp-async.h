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
 * hcpp-async.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef HCPP_ASYNC_H_
#define HCPP_ASYNC_H_

namespace hcpp {

/*
 * We just need to pack the function pointer and the pointer to
 * the heap allocated lambda. Due to this 16 bytes is sufficient
 * for our lambda based approach.
 */
#define MAX_HCPP_ASYNC_ARG_SIZE 16

template <typename Function, typename T1>
struct async_arguments1 {
	Function kernel;
	T1 arg1;

	async_arguments1(Function k, T1 a1) :
		kernel(k), arg1(a1) { }
};

template<typename Function, typename T1>
void wrapper1(void *args) {
	async_arguments1<Function, T1> *a =
			(async_arguments1<Function, T1> *)args;

	(*a->kernel)(a->arg1);
}

struct hcpp_async_task  {
	char _args[MAX_HCPP_ASYNC_ARG_SIZE];
	struct finish_t* current_finish;
	generic_framePtr _fp;
	bool is_asyncAnyType;
	ddf_t ** ddf_list; // Null terminated list
	inline void init_hcpp_async_task(generic_framePtr fp, size_t arg_sz, void *async_args) {
		HASSERT(arg_sz <= MAX_HCPP_ASYNC_ARG_SIZE);
		this->_fp = fp;
		this->is_asyncAnyType = false;
		this->ddf_list = NULL;
		memcpy(&this->_args, async_args, arg_sz);
	}

	inline void set_current_finish(struct finish_t* finish) {
		this->current_finish = finish;
	}

	inline struct finish_t* get_current_finish() {
		return this->current_finish;
	}

	inline void set_ddf_list(ddf_t ** ddf) {
		this->ddf_list = ddf;
	}

	inline ddf_t ** get_ddf_list() {
		return this->ddf_list;
	}

	inline void mark_as_asyncAnyTask() {
		this->is_asyncAnyType = true;
	}

	inline bool is_asyncAnyTask() {
		return this->is_asyncAnyType;
	}

	inline hcpp_async_task() { };

	template<typename Function, typename T1>
	inline hcpp_async_task(Function kernel, const T1 &a1) {
		async_arguments1<Function, T1> args(kernel, a1);
		init_hcpp_async_task(wrapper1<Function, T1>, (size_t)sizeof(args), (void *)&args);
	}
};

#define task_t 	hcpp_async_task

typedef struct hcpp_task_t {
	task_t async_task; 	// the actual task
	ddt_t ddt; 			// ddt meta-information
} hcpp_task_t;

void spawn(task_t * task);
void spawn_await(task_t * task, ddf_t** ddf_list);
void spawn_commTask(task_t * task);

#ifdef HUPCPP
#define MAX_REMOTE_ASYNCANY_ARG_SIZE 384

struct remoteAsyncAny_task  {
	generic_framePtr _fp;
	char _args[MAX_REMOTE_ASYNCANY_ARG_SIZE];
	inline void init_remoteAsyncAny_task(generic_framePtr fp, size_t arg_sz, void *remoteAsyncAny_args) {
		assert(arg_sz <= MAX_REMOTE_ASYNCANY_ARG_SIZE);
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
inline void execute_lambda(T* lambda) {
	const int wid = get_hc_wid();
	if(wid != 0) {
		// only computation workers can enter
		(*lambda)();
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

#else	 /* ! HUPCPP */
template <typename T>
inline void execute_lambda(T* lambda) {
	(*lambda)();
	HC_FREE((void*) lambda);
}
#endif	/* HUPCPP */

template <typename T>
inline task_t* _allocate_async(T lambda, bool await) {
	const size_t hcpp_task_size = !await ? sizeof(task_t) : sizeof(hcpp_task_t);
	task_t* task = (task_t*) HC_MALLOC(hcpp_task_size);
	const size_t lambda_size = sizeof(T);
	T* lambda_onHeap = (T*) HC_MALLOC(lambda_size);
	memcpy(lambda_onHeap, &lambda, lambda_size);
	task_t t = task_t(execute_lambda<T>, lambda_onHeap);
	memcpy(task, &t, sizeof(task_t));
	return task;
}

template <typename T>
inline void async(T lambda) {
	task_t* task = _allocate_async<T>(lambda, false);
	spawn(task);
}

template <typename T>
inline void _asyncAwait(ddf_t ** ddf_list, T lambda) {
	task_t* task = _allocate_async<T>(lambda, true);
	spawn_await(task, ddf_list);
}

template <typename T>
inline void asyncComm(T lambda) {
	task_t* task = _allocate_async<T>(lambda, false);
	spawn_commTask(task);
}

#ifdef HUPCPP
void spawn_asyncAnyTask(task_t * task);
template <typename T>
inline void asyncAny(T lambda) {
	task_t* task = _allocate_async<T>(lambda, false);
	task.mark_as_asyncAnyTask();
	spawn_asyncAnyTask(task);
}
#endif	/* HUPCPP */

}

#endif /* HCPP_ASYNC_H_ */
