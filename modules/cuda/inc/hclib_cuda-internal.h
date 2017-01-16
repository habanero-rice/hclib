#ifndef HCLIB_CUDA_INTERNAL_H
#define HCLIB_CUDA_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#ifdef HCLIB_INSTRUMENT
#include "hclib-instrument.h"
#endif

#include <cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

#define CHECK_CUDA(func) { \
    const cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(cuda_post_initialize);

int get_gpu_locale_id();
hclib::locale_t *get_closest_gpu_locale();
hclib::locale_t **get_gpu_locales(int *ngpus);
std::string get_gpu_name(hclib::locale_t *locale);
int get_cuda_device_id(hclib_locale_t *locale);
int get_num_gpu_locales();

#ifdef HCLIB_INSTRUMENT
int get_cuda_kernel_event_id();
#endif

}
#endif // HCLIB_CUDA_INTERNAL_H
