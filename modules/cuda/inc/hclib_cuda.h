#ifndef HCLIB_CUDA_H
#define HCLIB_CUDA_H

#include "hclib_cuda-internal.h"

namespace hclib {

template<typename functor_type>
inline __global__ void driver_kernel(functor_type functor) {
    functor();
}

template<class functor_type, typename... future_list_t>
inline hclib::future_t *forasync_cuda(const int blocks_per_gridx,
        const int blocks_per_gridy, const int blocks_per_gridz,
        const int threads_per_blockx, const int threads_per_blocky,
        const int threads_per_blockz, functor_type functor,
        hclib::locale_t *locale, hclib::future_t *future) {
    HASSERT(locale->type == get_gpu_locale_id());

    auto lambda = [locale, functor, blocks_per_gridx, blocks_per_gridy,
            blocks_per_gridz, threads_per_blockx, threads_per_blocky,
            threads_per_blockz] {
        CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

        dim3 blocks_per_grid(blocks_per_gridx, blocks_per_gridy,
                blocks_per_gridz);
        dim3 threads_per_block(threads_per_blockx, threads_per_blocky,
                threads_per_blockz);

        driver_kernel<<<blocks_per_grid, threads_per_block>>>(functor);
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    if (future) {
        return hclib::async_future_await_at(lambda, future, locale);
    } else {
        return hclib::async_future_at(lambda, locale);
    }
}

template<class functor_type, typename... future_list_t>
inline hclib::future_t *forasync_cuda(const int blocks_per_grid,
        const int threads_per_block, functor_type functor,
        hclib::locale_t *locale, hclib::future_t *future) {
    HASSERT(locale->type == get_gpu_locale_id());

    return forasync_cuda(blocks_per_grid, 1, 1, threads_per_block, 1, 1,
            functor, locale, future);
}

}

#endif
