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
 * ocr-runtime.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

#include "hcpp.h"
#include <sys/time.h>

namespace hcpp {
using namespace std;
#define ASYNC_COMM ((int) 0x2)

void execute_task(void* t) {
	task_t* task = (task_t*) t;
	(task->_fp)(task->_args);
	HC_FREE((void*) task);
}

void spawn(task_t * task) {
	::async(&execute_task, (void *) task, NO_DDF, NO_PHASER, NO_PROP);
}

void spawnComm(task_t * task) {
	::async(&execute_task, (void *) task, NO_DDF, NO_PHASER, ASYNC_COMM);
}

void spawn_await(task_t * task, ddf_t** ddf_list) {
	::async(&execute_task, (void *) task, ddf_list, NO_PHASER, NO_PROP);
}

void init(int * argc, char ** argv) {
	hclib_init(argc, argv);
}

void finalize() {
	hclib_finalize();
}

void start_finish() {
	::start_finish();
}

void end_finish() {
	::end_finish();
}

void finish(std::function<void()> lambda) {
	start_finish();
	lambda();
	end_finish();
}

int get_hc_wid() {
	return get_worker_id_hc();
}

int numWorkers() {
	return get_nb_workers();
}

}
