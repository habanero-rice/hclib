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
 * hcpp-asyncAwait.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 *      Acknowledgments: https://wiki.rice.edu/confluence/display/HABANERO/People
 */

/*
 * Following code generated from the perl file scripts/gen-generic-asyncAwait.pl
 *
 * Total DDFs supported in this version = 20. If you need to pass more than 20 DDFs
 * then generate template asyncAwait using the above perl file.
 */


#ifndef HCPP_ASYNCAWAIT_H_
#define HCPP_ASYNCAWAIT_H_

namespace hcpp {

template <typename T>
void asyncAwait(DDF_t* ddf0, T lambda) {
	int ddfs = 2;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, T lambda) {
	int ddfs = 3;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, T lambda) {
	int ddfs = 4;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, T lambda) {
	int ddfs = 5;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, T lambda) {
	int ddfs = 6;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, T lambda) {
	int ddfs = 7;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, T lambda) {
	int ddfs = 8;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, T lambda) {
	int ddfs = 9;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, T lambda) {
	int ddfs = 10;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, T lambda) {
	int ddfs = 11;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, T lambda) {
	int ddfs = 12;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, T lambda) {
	int ddfs = 13;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, T lambda) {
	int ddfs = 14;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, T lambda) {
	int ddfs = 15;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, T lambda) {
	int ddfs = 16;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, T lambda) {
	int ddfs = 17;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = ddf15;
	ddfList[16] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, T lambda) {
	int ddfs = 18;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = ddf15;
	ddfList[16] = ddf16;
	ddfList[17] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, T lambda) {
	int ddfs = 19;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = ddf15;
	ddfList[16] = ddf16;
	ddfList[17] = ddf17;
	ddfList[18] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, DDF_t* ddf18, T lambda) {
	int ddfs = 20;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = ddf15;
	ddfList[16] = ddf16;
	ddfList[17] = ddf17;
	ddfList[18] = ddf18;
	ddfList[19] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, DDF_t* ddf18, DDF_t* ddf19, T lambda) {
	int ddfs = 21;
	DDF_t** ddfList = (DDF_t**) HC_MALLOC(sizeof(DDF_t *) * ddfs);
	ddfList[0] = ddf0;
	ddfList[1] = ddf1;
	ddfList[2] = ddf2;
	ddfList[3] = ddf3;
	ddfList[4] = ddf4;
	ddfList[5] = ddf5;
	ddfList[6] = ddf6;
	ddfList[7] = ddf7;
	ddfList[8] = ddf8;
	ddfList[9] = ddf9;
	ddfList[10] = ddf10;
	ddfList[11] = ddf11;
	ddfList[12] = ddf12;
	ddfList[13] = ddf13;
	ddfList[14] = ddf14;
	ddfList[15] = ddf15;
	ddfList[16] = ddf16;
	ddfList[17] = ddf17;
	ddfList[18] = ddf18;
	ddfList[19] = ddf19;
	ddfList[20] = NULL;
	_asyncAwait<T>(ddfList, lambda);
}

}

#endif /* HCPP_ASYNCAWAIT_H_ */
