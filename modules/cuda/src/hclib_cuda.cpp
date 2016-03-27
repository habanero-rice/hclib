#include "hclib_cuda-internal.h"

#include <iostream>

static int gpu_locale_id;
static int gpus_so_far = 0;

int hclib::get_cuda_device_id(hclib_locale_t *locale) {
    return *((int *)(locale->metadata));
}

static void *allocation_func(size_t nbytes, hclib_locale_t *locale) {
    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaSetDevice(hclib::get_cuda_device_id(locale)));

    void *ptr;
    CHECK_CUDA(cudaMalloc((void **)&ptr, nbytes));
    return ptr;
}

static void free_func(void *ptr, hclib_locale_t *locale) {
    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaFree(ptr));
}

static void memset_func(void *ptr, int val, size_t nbytes,
        hclib_locale_t *locale) {
    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaMemset(ptr, val, nbytes));
}

static void copy_func(hclib::locale_t *dst_locale, void *dst,
        hclib::locale_t *src_locale, void *src, size_t nbytes) {
    cudaMemcpyKind kind;
    if (dst_locale->type == gpu_locale_id &&
            src_locale->type == gpu_locale_id) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (dst_locale->type == gpu_locale_id) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_locale->type == gpu_locale_id) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        HASSERT(false); // no CUDA device involved
    }
    CHECK_CUDA(cudaMemcpy(dst, src, nbytes, kind));
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

HCLIB_REGISTER_MODULE("cuda", cuda_pre_initialize, cuda_post_initialize, cuda_finalize)
