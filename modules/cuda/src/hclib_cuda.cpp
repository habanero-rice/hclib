#include "hclib_cuda-internal.h"

#include <iostream>

#ifdef HCLIB_INSTRUMENT
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

static int gpu_locale_id;
static int gpus_so_far = 0;

int hclib::get_cuda_device_id(hclib_locale_t *locale) {
    return *((int *)(locale->metadata));
}

static void *allocation_func(size_t nbytes, hclib_locale_t *locale) {
    CUDA_START_OP(ALLOC);

    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaSetDevice(hclib::get_cuda_device_id(locale)));

    void *ptr;
    CHECK_CUDA(cudaMalloc((void **)&ptr, nbytes));

    CUDA_END_OP(ALLOC);

    return ptr;
}

static void free_func(void *ptr, hclib_locale_t *locale) {
    CUDA_START_OP(FREE);

    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaFree(ptr));

    CUDA_END_OP(FREE);
}

static void memset_func(void *ptr, int val, size_t nbytes,
        hclib_locale_t *locale) {
    CUDA_START_OP(MEMSET);

    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaMemset(ptr, val, nbytes));

    CUDA_END_OP(MEMSET);
}

static void copy_func(hclib::locale_t *dst_locale, void *dst,
        hclib::locale_t *src_locale, void *src, size_t nbytes) {
    cudaMemcpyKind kind;
    if (dst_locale->type == gpu_locale_id &&
            src_locale->type == gpu_locale_id) {
        CUDA_START_OP(BETWEEN_DEVICE);
        kind = cudaMemcpyDeviceToDevice;
    } else if (dst_locale->type == gpu_locale_id) {
        CUDA_START_OP(TO_DEVICE);
        kind = cudaMemcpyHostToDevice;
    } else if (src_locale->type == gpu_locale_id) {
        CUDA_START_OP(FROM_DEVICE);
        kind = cudaMemcpyDeviceToHost;
    } else {
        HASSERT(false); // no CUDA device involved
    }

    const cudaError_t err = cudaMemcpy(dst, src, nbytes, kind);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR cudaMemcpy(dst=%p, src=%p, nbytes=%llu, "
                "kind=%s) - %s\n", dst, src, nbytes,
                (kind == cudaMemcpyDeviceToHost ? "cudaMemcpyDeviceToHost" :
                 (kind == cudaMemcpyHostToDevice ? "cudaMemcpyHostToDevice" :
                  "Unknown")), cudaGetErrorString(err));
        exit(1);
    }

#ifdef HCLIB_INSTRUMENT
    if (kind == cudaMemcpyDeviceToDevice) {
        CUDA_START_OP(BETWEEN_DEVICE);
    } else if (kind == cudaMemcpyHostToDevice) {
        CUDA_START_OP(TO_DEVICE);
    } else if (kind == cudaMemcpyDeviceToHost) {
        CUDA_START_OP(FROM_DEVICE);
    }
#endif
}

static size_t metadata_size() {
    return sizeof(int); // to store GPU ID
}

static void metadata_populate(hclib_locale_t *locale) {
    *((int *)locale->metadata) = gpus_so_far; // ID
    gpus_so_far++;
}

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_pre_initialize) {
    gpu_locale_id = hclib_add_known_locale_type("GPU");
    hclib_add_locale_metadata_functions(gpu_locale_id, metadata_size,
            metadata_populate);

#ifdef HCLIB_INSTRUMENT
    int i;
    for (i = 0; i < N_CUDA_FUNCS; i++) {
        event_ids[i] = register_event_type((char *)CUDA_FUNC_NAMES[i]);
    }
#endif
}

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_post_initialize) {
    hclib_register_alloc_func(gpu_locale_id, allocation_func);
    hclib_register_free_func(gpu_locale_id, free_func);
    hclib_register_memset_func(gpu_locale_id, memset_func);
    hclib_register_copy_func(gpu_locale_id, copy_func, MUST_USE);
}

HCLIB_MODULE_INITIALIZATION_FUNC(cuda_finalize) {
}

int hclib::get_gpu_locale_id() { return gpu_locale_id; }

hclib::locale_t *hclib::get_closest_gpu_locale() {
    return hclib_get_closest_locale_of_type(hclib_get_closest_locale(),
            gpu_locale_id);
}

hclib::locale_t **hclib::get_gpu_locales(int *ngpus) {
    return hclib_get_all_locales_of_type(gpu_locale_id, ngpus);
}

std::string hclib::get_gpu_name(hclib::locale_t *locale) {
    assert(locale->type == gpu_locale_id);
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, get_cuda_device_id(locale)));
    return std::string(prop.name);
}

int hclib::get_num_gpu_locales() {
    return hclib_get_num_locales_of_type(gpu_locale_id);
}

#ifdef HCLIB_INSTRUMENT
int hclib::get_cuda_kernel_event_id() {
    return event_ids[KERNEL_lbl];
}
#endif

HCLIB_REGISTER_MODULE("cuda", cuda_pre_initialize, cuda_post_initialize, cuda_finalize)
