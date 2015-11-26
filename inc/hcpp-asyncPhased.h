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
 * hcpp-asyncPhased.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Following code generated from the perl file scripts/gen-generic-asyncPhased.pl
 *
 * Total phasers supported in this version = 20. If you need to pass more than 20,
 * then generate template asyncPhased using the above perl file.
 */


#ifndef HCPP_ASYNCPHASED_H_
#define HCPP_ASYNCPHASED_H_

#ifdef _PHASERS_
namespace hcpp {

template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , T lambda) {
	int total = 1;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, T lambda) {
	int total = 2;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, T lambda) {
	int total = 3;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, T lambda) {
	int total = 4;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, T lambda) {
	int total = 5;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, T lambda) {
	int total = 6;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, T lambda) {
	int total = 7;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, T lambda) {
	int total = 8;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, T lambda) {
	int total = 9;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, T lambda) {
	int total = 10;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, T lambda) {
	int total = 11;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, T lambda) {
	int total = 12;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, T lambda) {
	int total = 13;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, T lambda) {
	int total = 14;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, T lambda) {
	int total = 15;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, PHASER_t ph15, PHASER_m m15, T lambda) {
	int total = 16;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phaser_type_arr[15] = ph15;
	phaser_mode_arr[15] = m15;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, PHASER_t ph15, PHASER_m m15, PHASER_t ph16, PHASER_m m16, T lambda) {
	int total = 17;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phaser_type_arr[15] = ph15;
	phaser_mode_arr[15] = m15;
	phaser_type_arr[16] = ph16;
	phaser_mode_arr[16] = m16;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, PHASER_t ph15, PHASER_m m15, PHASER_t ph16, PHASER_m m16, PHASER_t ph17, PHASER_m m17, T lambda) {
	int total = 18;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phaser_type_arr[15] = ph15;
	phaser_mode_arr[15] = m15;
	phaser_type_arr[16] = ph16;
	phaser_mode_arr[16] = m16;
	phaser_type_arr[17] = ph17;
	phaser_mode_arr[17] = m17;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, PHASER_t ph15, PHASER_m m15, PHASER_t ph16, PHASER_m m16, PHASER_t ph17, PHASER_m m17, PHASER_t ph18, PHASER_m m18, T lambda) {
	int total = 19;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phaser_type_arr[15] = ph15;
	phaser_mode_arr[15] = m15;
	phaser_type_arr[16] = ph16;
	phaser_mode_arr[16] = m16;
	phaser_type_arr[17] = ph17;
	phaser_mode_arr[17] = m17;
	phaser_type_arr[18] = ph18;
	phaser_mode_arr[18] = m18;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}
template <typename T>
void asyncPhased(PHASER_t ph0, PHASER_m m0 , PHASER_t ph1, PHASER_m m1, PHASER_t ph2, PHASER_m m2, PHASER_t ph3, PHASER_m m3, PHASER_t ph4, PHASER_m m4, PHASER_t ph5, PHASER_m m5, PHASER_t ph6, PHASER_m m6, PHASER_t ph7, PHASER_m m7, PHASER_t ph8, PHASER_m m8, PHASER_t ph9, PHASER_m m9, PHASER_t ph10, PHASER_m m10, PHASER_t ph11, PHASER_m m11, PHASER_t ph12, PHASER_m m12, PHASER_t ph13, PHASER_m m13, PHASER_t ph14, PHASER_m m14, PHASER_t ph15, PHASER_m m15, PHASER_t ph16, PHASER_m m16, PHASER_t ph17, PHASER_m m17, PHASER_t ph18, PHASER_m m18, PHASER_t ph19, PHASER_m m19, T lambda) {
	int total = 20;
	PHASER_t* phaser_type_arr = (PHASER_t*) HC_MALLOC(sizeof(PHASER_t) * total);
	PHASER_m* phaser_mode_arr = (PHASER_m*) HC_MALLOC(sizeof(PHASER_m) * total);
	phaser_type_arr[0] = ph0;
	phaser_mode_arr[0] = m0;
	phaser_type_arr[1] = ph1;
	phaser_mode_arr[1] = m1;
	phaser_type_arr[2] = ph2;
	phaser_mode_arr[2] = m2;
	phaser_type_arr[3] = ph3;
	phaser_mode_arr[3] = m3;
	phaser_type_arr[4] = ph4;
	phaser_mode_arr[4] = m4;
	phaser_type_arr[5] = ph5;
	phaser_mode_arr[5] = m5;
	phaser_type_arr[6] = ph6;
	phaser_mode_arr[6] = m6;
	phaser_type_arr[7] = ph7;
	phaser_mode_arr[7] = m7;
	phaser_type_arr[8] = ph8;
	phaser_mode_arr[8] = m8;
	phaser_type_arr[9] = ph9;
	phaser_mode_arr[9] = m9;
	phaser_type_arr[10] = ph10;
	phaser_mode_arr[10] = m10;
	phaser_type_arr[11] = ph11;
	phaser_mode_arr[11] = m11;
	phaser_type_arr[12] = ph12;
	phaser_mode_arr[12] = m12;
	phaser_type_arr[13] = ph13;
	phaser_mode_arr[13] = m13;
	phaser_type_arr[14] = ph14;
	phaser_mode_arr[14] = m14;
	phaser_type_arr[15] = ph15;
	phaser_mode_arr[15] = m15;
	phaser_type_arr[16] = ph16;
	phaser_mode_arr[16] = m16;
	phaser_type_arr[17] = ph17;
	phaser_mode_arr[17] = m17;
	phaser_type_arr[18] = ph18;
	phaser_mode_arr[18] = m18;
	phaser_type_arr[19] = ph19;
	phaser_mode_arr[19] = m19;
	phased_t phased;
	phased.count = total;
	phased.phasers = phaser_type_arr;
	phased.phasers_mode = phaser_mode_arr;
	_asyncPhased<T>(&phased, lambda);
}

}

#endif // _PHASERS_
#endif /* HCPP_ASYNCPHASED_H_ */
