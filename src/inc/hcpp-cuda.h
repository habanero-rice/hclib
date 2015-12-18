#ifndef _HCPP_CUDA_H
#define _HCPP_CUDA_H

#ifdef HC_CUDA

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "hcpp-place.h"

#define CHECK_CUDA(call) { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

/*
 * There are two different types of GPU tasks we can run. A GPU_COMM_TASK maps
 * down to a cudaMemcpy/memcpy. A GPU_COMPUTE_TASK maps down to a GPU kernel
 * launch.
 */
typedef enum {
    GPU_COMM_TASK,
    GPU_MEMSET_TASK,
    GPU_COMPUTE_TASK
} gpu_task_type_t;

/*
 * The definition of a GPU communication task, including:
 *   1. The source and destination places. These can specify a communication of
 *      type host->device, device->host, host->host, or device->device.
 *   2. Pointers in src_pl and dst_pl that store the actual memory locations to
 *      copy to and from.
 *   3. Number of bytes to transfer between src_pl and dst_pl.
 */
typedef struct _gpu_comm_task_t {
    place_t *src_pl, *dst_pl;
    void *src, *dst;
    size_t nbytes;
} gpu_comm_task_t;

/*
 * Definition of a task that clears the target buffer to val.
 */
typedef struct _gpu_memset_task_t {
    place_t *pl;
    void *ptr;
    int val;
    size_t nbytes;
} gpu_memset_task_t;

/*
 * The definition of a GPU compute task. TODO Currently incomplete.
 */
typedef struct _gpu_functor_wrapper {
    void (*functor_caller)(unsigned, unsigned, void *);
    void *functor_on_heap;
} gpu_functor_wrapper;

typedef struct _gpu_compute_task_t {
    unsigned niters, tile_size;
    int cuda_id;
    gpu_functor_wrapper *kernel_launcher;
} gpu_compute_task_t;

/*
 * A GPU task that can be placed in and popped from a regular deque_t or
 * semi_conc_deque_t.
 */
typedef struct _gpu_task_t {
    task_t t;
    gpu_task_type_t gpu_type;
    // DDF to put when this task completes
    hclib_ddf_t *ddf_to_put;
    void *arg_to_put;
    union {
        gpu_comm_task_t comm_task;
        gpu_memset_task_t memset_task;
        gpu_compute_task_t compute_task;
    } gpu_task_def;
} gpu_task_t;

/*
 * pending_cuda_op represents an asynchronous CUDA operation that has been
 * kicked off and which will signal event when it completes. pending_cuda_op is
 * used for user GPU copy or kernel tasks that are launched with some dependent
 * DDFs that should be put following the completion of the task. Rather than
 * block the whole GPU thread on each CUDA operation, we can queue up many
 * asynchronous CUDA operations but need somewhere to track them all.
 * pending_cuda_op provides this tracking.
 */
typedef struct _pending_cuda_op {
    hclib_ddf_t *ddf_to_put;
    void *arg_to_put;
    cudaEvent_t event;
    struct _pending_cuda_op *next;
} pending_cuda_op;

#endif

#endif
