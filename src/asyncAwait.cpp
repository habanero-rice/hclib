#include "hcpp-utils.h"
namespace hcpp { 
  using namespace std; 
  void asyncAwait(DDF_t* ddf0, std::function<void()> &&lambda) {
    int ddfs = 2;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, std::function<void()> &&lambda) {
    int ddfs = 3;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, std::function<void()> &&lambda) {
    int ddfs = 4;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, std::function<void()> &&lambda) {
    int ddfs = 5;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = ddf3; 
    ddfList[4] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, std::function<void()> &&lambda) {
    int ddfs = 6;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = ddf3; 
    ddfList[4] = ddf4; 
    ddfList[5] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, std::function<void()> &&lambda) {
    int ddfs = 7;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = ddf3; 
    ddfList[4] = ddf4; 
    ddfList[5] = ddf5; 
    ddfList[6] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, std::function<void()> &&lambda) {
    int ddfs = 8;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = ddf3; 
    ddfList[4] = ddf4; 
    ddfList[5] = ddf5; 
    ddfList[6] = ddf6; 
    ddfList[7] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, std::function<void()> &&lambda) {
    int ddfs = 9;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
    ddfList[0] = ddf0; 
    ddfList[1] = ddf1; 
    ddfList[2] = ddf2; 
    ddfList[3] = ddf3; 
    ddfList[4] = ddf4; 
    ddfList[5] = ddf5; 
    ddfList[6] = ddf6; 
    ddfList[7] = ddf7; 
    ddfList[8] = NULL; 
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, std::function<void()> &&lambda) {
    int ddfs = 10;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, std::function<void()> &&lambda) {
    int ddfs = 11;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, std::function<void()> &&lambda) {
    int ddfs = 12;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, std::function<void()> &&lambda) {
    int ddfs = 13;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, std::function<void()> &&lambda) {
    int ddfs = 14;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, std::function<void()> &&lambda) {
    int ddfs = 15;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, std::function<void()> &&lambda) {
    int ddfs = 16;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, std::function<void()> &&lambda) {
    int ddfs = 17;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, std::function<void()> &&lambda) {
    int ddfs = 18;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, std::function<void()> &&lambda) {
    int ddfs = 19;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, DDF_t* ddf18, std::function<void()> &&lambda) {
    int ddfs = 20;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
  void asyncAwait(DDF_t* ddf0, DDF_t* ddf1, DDF_t* ddf2, DDF_t* ddf3, DDF_t* ddf4, DDF_t* ddf5, DDF_t* ddf6, DDF_t* ddf7, DDF_t* ddf8, DDF_t* ddf9, DDF_t* ddf10, DDF_t* ddf11, DDF_t* ddf12, DDF_t* ddf13, DDF_t* ddf14, DDF_t* ddf15, DDF_t* ddf16, DDF_t* ddf17, DDF_t* ddf18, DDF_t* ddf19, std::function<void()> &&lambda) {
    int ddfs = 21;
    DDF_t** ddfList = (DDF_t**) malloc(sizeof(DDF_t *) * ddfs);
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
    std::function<void()> * copy_of_lambda = new std::function<void()> (lambda);
    ::async(&async_cpp_wrapper, (void *)copy_of_lambda, ddfList, NO_PHASER, NO_PROP);
  }
}
