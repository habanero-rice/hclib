#include "hclib-ddf.h"
#include "hclib-async.h"

namespace hclib {

template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, T lambda) {
	int ddfs = 2;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, T lambda) {
	int ddfs = 3;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, T lambda) {
	int ddfs = 4;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, T lambda) {
	int ddfs = 5;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = ddf3; 
	ddfList[4] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, T lambda) {
	int ddfs = 6;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = ddf3; 
	ddfList[4] = ddf4; 
	ddfList[5] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, T lambda) {
	int ddfs = 7;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = ddf3; 
	ddfList[4] = ddf4; 
	ddfList[5] = ddf5; 
	ddfList[6] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, T lambda) {
	int ddfs = 8;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = ddf3; 
	ddfList[4] = ddf4; 
	ddfList[5] = ddf5; 
	ddfList[6] = ddf6; 
	ddfList[7] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, T lambda) {
	int ddfs = 9;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
	ddfList[0] = ddf0; 
	ddfList[1] = ddf1; 
	ddfList[2] = ddf2; 
	ddfList[3] = ddf3; 
	ddfList[4] = ddf4; 
	ddfList[5] = ddf5; 
	ddfList[6] = ddf6; 
	ddfList[7] = ddf7; 
	ddfList[8] = NULL; 
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, T lambda) {
	int ddfs = 10;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, T lambda) {
	int ddfs = 11;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, T lambda) {
	int ddfs = 12;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, T lambda) {
	int ddfs = 13;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, T lambda) {
	int ddfs = 14;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, T lambda) {
	int ddfs = 15;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, T lambda) {
	int ddfs = 16;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, hclib_ddf_t* ddf15, T lambda) {
	int ddfs = 17;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, hclib_ddf_t* ddf15, hclib_ddf_t* ddf16, T lambda) {
	int ddfs = 18;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, hclib_ddf_t* ddf15, hclib_ddf_t* ddf16, hclib_ddf_t* ddf17, T lambda) {
	int ddfs = 19;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, hclib_ddf_t* ddf15, hclib_ddf_t* ddf16, hclib_ddf_t* ddf17, hclib_ddf_t* ddf18, T lambda) {
	int ddfs = 20;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}
template <typename T>
void asyncAwait(hclib_ddf_t* ddf0, hclib_ddf_t* ddf1, hclib_ddf_t* ddf2, hclib_ddf_t* ddf3, hclib_ddf_t* ddf4, hclib_ddf_t* ddf5, hclib_ddf_t* ddf6, hclib_ddf_t* ddf7, hclib_ddf_t* ddf8, hclib_ddf_t* ddf9, hclib_ddf_t* ddf10, hclib_ddf_t* ddf11, hclib_ddf_t* ddf12, hclib_ddf_t* ddf13, hclib_ddf_t* ddf14, hclib_ddf_t* ddf15, hclib_ddf_t* ddf16, hclib_ddf_t* ddf17, hclib_ddf_t* ddf18, hclib_ddf_t* ddf19, T lambda) {
	int ddfs = 21;
	hclib_ddf_t** ddfList = (hclib_ddf_t**) HC_MALLOC(sizeof(hclib_ddf_t *) * ddfs);
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
	hclib::_asyncAwait<T>(ddfList, lambda);
}

}

