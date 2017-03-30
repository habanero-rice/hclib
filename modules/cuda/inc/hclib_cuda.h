#ifndef HCLIB_CUDA_H
#define HCLIB_CUDA_H

#include "hclib_cuda-internal.h"
#include "hclib-module-common.h"

namespace hclib {

template<typename functor_type>
__global__ void driver_kernel(functor_type functor) {
    functor();
}

template<typename functor_type>
__global__ void simd_driver_kernel(functor_type functor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = gridDim.x * blockDim.x;
    functor(tid, width);
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
inline hclib::future_t<void> *async_simd(functor_type functor, hclib::locale_t *locale) {
    CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

    cudaStream_t stream = get_stream(locale);
    hclib::promise_t<void> *prom = new hclib::promise_t<void>();

    cudaEvent_t event;
    CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    simd_driver_kernel<<<1, 32, 0, stream>>>(functor);
    CHECK_CUDA(cudaEventRecord(event, stream));

    pending_cuda_op *op = (pending_cuda_op *)malloc(sizeof(*op));
    assert(op);
    op->event = event;
    op->prom = prom;
    op->task = NULL;
    hclib::append_to_pending(op, &pending_cuda, hclib::test_cuda_completion, locale);

    return prom->get_future();
}

}

#endif
