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

#include "hcpp-asyncStruct.h"
#include "hcupc-support.h"

#ifndef HCPP_ASYNC_H_
#define HCPP_ASYNC_H_

namespace hcpp {

template <typename T>
inline void execute_hcpp_lambda(T* lambda) {
	const int wid = current_ws()->id;
	MARK_BUSY(wid);
	(*lambda)();
	HC_FREE((void*) lambda);
	MARK_OVH(wid);
}

template <typename T>
inline task_t* _allocate_async_hcpp(T lambda, bool await) {
	const size_t hcpp_task_size = !await ? sizeof(task_t) : sizeof(hcpp_task_t);
	task_t* task = (task_t*) HC_MALLOC(hcpp_task_size);
	const size_t lambda_size = sizeof(T);
	T* lambda_onHeap = (T*) HC_MALLOC(lambda_size);
	memcpy(lambda_onHeap, &lambda, lambda_size);
	task_t t = task_t(execute_hcpp_lambda<T>, lambda_onHeap);
	memcpy(task, &t, sizeof(task_t));
	return task;
}

#if defined(HUPCPP) && defined(DIST_WS)	// i.e. if we are supporting distributed work-stealing in HabaneroUPC++
#define _allocate_async _allocate_async_hcupc
#else
#define _allocate_async _allocate_async_hcpp
#endif

/*
 * Yes, the name "asyncAtHpt" sounds weird
 * but using this name to differentiate with the inter-node
 * "asyncAt" in HabaneroUPC++. Feel free to give a better
 * name to asyncAtHpt.
 */
template <typename T>
inline void asyncAtHpt(place_t* pl, T lambda) {
	MARK_OVH(current_ws()->id);
	task_t* task = _allocate_async<T>(lambda, false);
	spawn_at_hpt(pl, task);
}

template <typename T>
inline void async(T lambda) {
	MARK_OVH(current_ws()->id);
	task_t* task = _allocate_async<T>(lambda, false);
	spawn(task);
}

template <typename T>
inline void _asyncAwait(ddf_t ** ddf_list, T lambda) {
	MARK_OVH(current_ws()->id);
	task_t* task = _allocate_async<T>(lambda, true);
	spawn_await(task, ddf_list);
}

template <typename T>
inline void asyncComm(T lambda) {
	task_t* task = _allocate_async<T>(lambda, false);
	spawn_commTask(task);
}

}

#endif /* HCPP_ASYNC_H_ */

