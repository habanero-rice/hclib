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
inline void isolated(void* object0, void* object1, std::function<void()> &&lambda) {
	const int n = 2;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, std::function<void()> &&lambda) {
	const int n = 3;
	void *array[n];
	array[0] = object0; 
	array[1] = object1; 
	array[2] = object2; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
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
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, std::function<void()> &&lambda) {
	const int n = 11;
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
	array[10] = object10; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, std::function<void()> &&lambda) {
	const int n = 12;
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
	array[10] = object10; 
	array[11] = object11; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, std::function<void()> &&lambda) {
	const int n = 13;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, std::function<void()> &&lambda) {
	const int n = 14;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, std::function<void()> &&lambda) {
	const int n = 15;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, void* object15, std::function<void()> &&lambda) {
	const int n = 16;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	array[15] = object15; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, void* object15, void* object16, std::function<void()> &&lambda) {
	const int n = 17;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	array[15] = object15; 
	array[16] = object16; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, void* object15, void* object16, void* object17, std::function<void()> &&lambda) {
	const int n = 18;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	array[15] = object15; 
	array[16] = object16; 
	array[17] = object17; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, void* object15, void* object16, void* object17, void* object18, std::function<void()> &&lambda) {
	const int n = 19;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	array[15] = object15; 
	array[16] = object16; 
	array[17] = object17; 
	array[18] = object18; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}
inline void isolated(void* object0, void* object1, void* object2, void* object3, void* object4, void* object5, void* object6, void* object7, void* object8, void* object9, void* object10, void* object11, void* object12, void* object13, void* object14, void* object15, void* object16, void* object17, void* object18, void* object19, std::function<void()> &&lambda) {
	const int n = 20;
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
	array[10] = object10; 
	array[11] = object11; 
	array[12] = object12; 
	array[13] = object13; 
	array[14] = object14; 
	array[15] = object15; 
	array[16] = object16; 
	array[17] = object17; 
	array[18] = object18; 
	array[19] = object19; 
	isolated_execution(array, n, execute_isolation_lambda, (void*)&lambda);
}

}
#endif
