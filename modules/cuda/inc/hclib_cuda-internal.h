#ifndef HCLIB_CUDA_INTERNAL_H
#define HCLIB_CUDA_INTERNAL_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

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

#ifdef HCLIB_INSTRUMENT
#include "hclib-instrument.h"

enum CUDA_FUNC_LABELS {
    TO_DEVICE_lbl = 0,
    FROM_DEVICE_lbl,
    BETWEEN_DEVICE_lbl,
    KERNEL_lbl,
    MEMSET_lbl,
    ALLOC_lbl,
    FREE_lbl,
    N_CUDA_FUNCS
};

const char *CUDA_FUNC_NAMES[N_CUDA_FUNCS] = {
    "TO_DEVICE",
    "FROM_DEVICE",
    "BETWEEN_DEVICE",
    "KERNEL",
    "MEMSET",
    "ALLOC",
    "FREE"
};

static int event_ids[N_CUDA_FUNCS];

#define CUDA_START_OP(funcname) \
    const unsigned _event_id = hclib_register_event(event_ids[funcname##_lbl], \
            START, -1)
#define CUDA_END_OP(funcname) \
    hclib_register_event(event_ids[funcname##_lbl], END, _event_id)

#else
#define CUDA_START_OP(funcname)
#define CUDA_END_OP(funcname)
#endif


namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(cuda_post_initialize);

int get_gpu_locale_id();
hclib::locale_t *get_closest_gpu_locale();
hclib::locale_t **get_gpu_locales(int *ngpus);
std::string get_gpu_name(hclib::locale_t *locale);
int get_cuda_device_id(hclib_locale_t *locale);
int get_num_gpu_locales();

}

#endif
