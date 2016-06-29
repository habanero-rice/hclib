/* Copyright (c) 2016, Rice University

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


#ifndef HCLIB_ISOLATED_OVERLOADED_H_
#define HCLIB_ISOLATED_OVERLOADED_H_

/*
 * hclib-isolated.h
 *
 *      Authors: Vivek Kumar (vivekk@rice.edu)
 *
 * Other than the first function "execute_isolation_lambda", rest all
 * inline functions are generated using the perl script: scripts/gen-generic-isolated.pl 
 * $./scripts/gen-generic-isolated.pl 10
 * 
 */

namespace hclib {

inline void execute_isolation_lambda(void * args) {
  std::function<void()> *lambda = (std::function<void()> *)args;
  (*lambda)();
}

inline void isolated(void* object0, std::function<void()> &&lambda) {
	const int n = 1;
	void *array[n];
	array[0] = object0; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0) {
	const int n = 1;
	void *array[n];
	array[0] = object0; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0) {
	const int n = 1;
	void *array[n];
	array[0] = object0; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, std::function<void()> &&lambda) {
	const int n = 2;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1) {
	const int n = 2;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1) {
	const int n = 2;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, std::function<void()> &&lambda) {
	const int n = 3;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2) {
	const int n = 3;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2) {
	const int n = 3;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, std::function<void()> &&lambda) {
	const int n = 4;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3) {
	const int n = 4;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3) {
	const int n = 4;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, std::function<void()> &&lambda) {
	const int n = 5;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4) {
	const int n = 5;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4) {
	const int n = 5;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, std::function<void()> &&lambda) {
	const int n = 6;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5) {
	const int n = 6;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5) {
	const int n = 6;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, std::function<void()> &&lambda) {
	const int n = 7;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6) {
	const int n = 7;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6) {
	const int n = 7;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, std::function<void()> &&lambda) {
	const int n = 8;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7) {
	const int n = 8;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7) {
	const int n = 8;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, std::function<void()> &&lambda) {
	const int n = 9;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8) {
	const int n = 9;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8) {
	const int n = 9;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	remove_isolation(array, n);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, std::function<void()> &&lambda) {
	const int n = 10;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	array[9] = object9; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void enable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9) {
	const int n = 10;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	array[9] = object9; 
	apply_isolation(array, n);
}
inline void disable_isolation(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9) {
	const int n = 10;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	array[3] = object3; 
	array[4] = object4; 
	array[5] = object5; 
	array[6] = object6; 
	array[7] = object7; 
	array[8] = object8; 
	array[9] = object9; 
	remove_isolation(array, n);
}

}
#endif
