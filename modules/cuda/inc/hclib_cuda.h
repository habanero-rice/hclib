#ifndef HCLIB_CUDA_H
#define HCLIB_CUDA_H

#include "hclib_cuda-internal.h"

namespace hclib {

template<typename functor_type>
inline __global__ void driver_kernel(functor_type functor, unsigned niters) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < niters) {
        functor(tid);
    }   
}

template<class functor_type, typename... future_list_t>
inline hclib::future_t *forasync_cuda(int niters, functor_type functor,
        hclib::locale_t *locale, future_list_t... futures) {
    HASSERT(locale->type == get_gpu_locale_id());

    return hclib::async_future_await_at([locale, functor, niters] {
        CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

        const int threads_per_block = 256;
        const int blocks_per_grid = (niters + threads_per_block - 1) /
            threads_per_block;

        driver_kernel<<<blocks_per_grid, threads_per_block>>>(functor, niters);
        CHECK_CUDA(cudaDeviceSynchronize());
    }, locale, futures...);
}

}

#endif
