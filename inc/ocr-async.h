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
 * ocr-async.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#ifndef OCR_ASYNC_H_
#define OCR_ASYNC_H_

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
	generic_framePtr _fp;
	inline void init_hcpp_async_task(generic_framePtr fp, size_t arg_sz, void *async_args) {
		HASSERT(arg_sz <= MAX_HCPP_ASYNC_ARG_SIZE);
		this->_fp = fp;
		memcpy(&this->_args, async_args, arg_sz);
	}

	inline hcpp_async_task() { };

	template<typename Function, typename T1>
	inline hcpp_async_task(Function kernel, const T1 &a1) {
		async_arguments1<Function, T1> args(kernel, a1);
		init_hcpp_async_task(wrapper1<Function, T1>, (size_t)sizeof(args), (void *)&args);
	}
};

#define task_t 	hcpp_async_task

void spawn(task_t * task);
void spawnComm(task_t * task);
void spawn_await(task_t * task, ddf_t** ddf_list);

template <typename T>
inline void execute_lambda(T* lambda) {
	(*lambda)();
	HC_FREE((void*) lambda);
}

template <typename T>
inline task_t* _allocate_async(T lambda) {
	const size_t hcpp_task_size = sizeof(task_t);
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
	task_t* task = _allocate_async<T>(lambda);
	spawn(task);
}

template <typename T>
inline void asyncComm(T lambda) {
	task_t* task = _allocate_async<T>(lambda);
	spawnComm(task);
}

template <typename T>
inline void _asyncAwait(ddf_t ** ddf_list, T lambda) {
	task_t* task = _allocate_async<T>(lambda);
	spawn_await(task, ddf_list);
}

}

#endif /* OCR_ASYNC_H_ */
