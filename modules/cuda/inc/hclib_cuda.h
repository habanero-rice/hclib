#ifndef HCLIB_CUDA_H
#define HCLIB_CUDA_H

#include "hclib_cuda-internal.h"

namespace hclib {

template<typename functor_type>
__global__ void driver_kernel(functor_type functor) {
    functor();
}

template<class functor_type, class await_type>
inline hclib::future_t<void> *forasync_cuda(const int blocks_per_gridx,
        const int blocks_per_gridy, const int blocks_per_gridz,
        const int threads_per_blockx, const int threads_per_blocky,
        const int threads_per_blockz, const int shared_mem,
        functor_type functor, hclib::locale_t *locale,
        hclib::future_t<await_type> *future) {
    HASSERT(locale->type == get_gpu_locale_id());

    return hclib::async_future_await_at([locale, functor, blocks_per_gridx, blocks_per_gridy,
            blocks_per_gridz, threads_per_blockx, threads_per_blocky,
            threads_per_blockz, shared_mem] {
#ifdef HCLIB_INSTRUMENT
    const unsigned _event_id = hclib_register_event(get_cuda_kernel_event_id(),
            START, -1);
#endif

        CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

        dim3 blocks_per_grid(blocks_per_gridx, blocks_per_gridy,
                blocks_per_gridz);
        dim3 threads_per_block(threads_per_blockx, threads_per_blocky,
                threads_per_blockz);

        driver_kernel<<<blocks_per_grid, threads_per_block, shared_mem>>>(functor);
        CHECK_CUDA(cudaDeviceSynchronize());

#ifdef HCLIB_INSTRUMENT
        hclib_register_event(get_cuda_kernel_event_id(), END, _event_id);
#endif
    }, future, locale);
}

template<class functor_type, class await_type>
inline hclib::future_t<void> *forasync_cuda(const int blocks_per_gridx,
        const int blocks_per_gridy, const int blocks_per_gridz,
        const int threads_per_blockx, const int threads_per_blocky,
        const int threads_per_blockz,
        functor_type functor, hclib::locale_t *locale,
        hclib::future_t<await_type> *future) {
    return forasync_cuda(blocks_per_gridx, blocks_per_gridy, blocks_per_gridz,
            threads_per_blockx, threads_per_blocky, threads_per_blockz, 0,
            functor, locale, future);
}

template<class functor_type, class await_type>
inline hclib::future_t<void> *forasync_cuda(const int blocks_per_grid,
        const int threads_per_block, functor_type functor,
        hclib::locale_t *locale, hclib::future_t<await_type> *future) {
    return forasync_cuda(blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0,
            functor, locale, future);
}

template<class functor_type, class await_type>
inline hclib::future_t<void> *forasync_cuda(const int blocks_per_grid,
        const int threads_per_block, const int shared_mem, functor_type functor,
        hclib::locale_t *locale, hclib::future_t<await_type> *future) {
    return forasync_cuda(blocks_per_grid, 1, 1, threads_per_block, 1, 1,
            shared_mem, functor, locale, future);
}

template<class functor_type>
inline void async_simd(functor_type functor, hclib::locale_t *locale) {
    CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

    driver_kernel<<<1, 32, 0, get_stream(locale)>>>(functor);
}

}

#endif
