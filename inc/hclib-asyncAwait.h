#include "hclib-promise.h"
#include "hclib-async.h"

namespace hclib {

template <typename T>
void asyncAwait(hclib_promise_t* promise0, T lambda) {
	int promises = 2;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, T lambda) {
	int promises = 3;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, T lambda) {
	int promises = 4;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, T lambda) {
	int promises = 5;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, T lambda) {
	int promises = 6;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, T lambda) {
	int promises = 7;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, T lambda) {
	int promises = 8;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, T lambda) {
	int promises = 9;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, T lambda) {
	int promises = 10;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, T lambda) {
	int promises = 11;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, T lambda) {
	int promises = 12;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, T lambda) {
	int promises = 13;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, T lambda) {
	int promises = 14;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, T lambda) {
	int promises = 15;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, T lambda) {
	int promises = 16;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, hclib_promise_t* promise15, T lambda) {
	int promises = 17;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = promise15; 
	promiseList[16] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, hclib_promise_t* promise15, hclib_promise_t* promise16, T lambda) {
	int promises = 18;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = promise15; 
	promiseList[16] = promise16; 
	promiseList[17] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, hclib_promise_t* promise15, hclib_promise_t* promise16, hclib_promise_t* promise17, T lambda) {
	int promises = 19;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = promise15; 
	promiseList[16] = promise16; 
	promiseList[17] = promise17; 
	promiseList[18] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, hclib_promise_t* promise15, hclib_promise_t* promise16, hclib_promise_t* promise17, hclib_promise_t* promise18, T lambda) {
	int promises = 20;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = promise15; 
	promiseList[16] = promise16; 
	promiseList[17] = promise17; 
	promiseList[18] = promise18; 
	promiseList[19] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}
template <typename T>
void asyncAwait(hclib_promise_t* promise0, hclib_promise_t* promise1, hclib_promise_t* promise2, hclib_promise_t* promise3, hclib_promise_t* promise4, hclib_promise_t* promise5, hclib_promise_t* promise6, hclib_promise_t* promise7, hclib_promise_t* promise8, hclib_promise_t* promise9, hclib_promise_t* promise10, hclib_promise_t* promise11, hclib_promise_t* promise12, hclib_promise_t* promise13, hclib_promise_t* promise14, hclib_promise_t* promise15, hclib_promise_t* promise16, hclib_promise_t* promise17, hclib_promise_t* promise18, hclib_promise_t* promise19, T lambda) {
	int promises = 21;
	hclib_promise_t** promiseList = (hclib_promise_t**) HC_MALLOC(sizeof(hclib_promise_t *) * promises);
	promiseList[0] = promise0; 
	promiseList[1] = promise1; 
	promiseList[2] = promise2; 
	promiseList[3] = promise3; 
	promiseList[4] = promise4; 
	promiseList[5] = promise5; 
	promiseList[6] = promise6; 
	promiseList[7] = promise7; 
	promiseList[8] = promise8; 
	promiseList[9] = promise9; 
	promiseList[10] = promise10; 
	promiseList[11] = promise11; 
	promiseList[12] = promise12; 
	promiseList[13] = promise13; 
	promiseList[14] = promise14; 
	promiseList[15] = promise15; 
	promiseList[16] = promise16; 
	promiseList[17] = promise17; 
	promiseList[18] = promise18; 
	promiseList[19] = promise19; 
	promiseList[20] = NULL; 
	hclib::_asyncAwait<T>(promiseList, lambda);
}

}

