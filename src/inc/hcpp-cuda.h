#ifndef _HCPP_CUDA_H
#define _HCPP_CUDA_H

#ifdef HC_CUDA

#include <driver_types.h>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(call) { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#endif

#endif
