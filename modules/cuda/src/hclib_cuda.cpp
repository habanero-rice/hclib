#include "hclib_cuda.h"

#include <iostream>

static int gpu_locale_id;
static int gpus_so_far = 0;

static int get_cuda_device_id(hclib_locale_t *locale) {
    return *((int *)(locale->metadata));
}

static void *allocation_func(size_t nbytes, hclib_locale_t *locale) {
    assert(locale->type == gpu_locale_id);
    CHECK_CUDA(cudaSetDevice(get_cuda_device_id(locale)));

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
}

int hclib::get_gpu_locale_id() { return gpu_locale_id; }

hclib::locale_t *hclib::get_closest_gpu_locale() {
    return hclib_get_closest_locale_of_type(hclib_get_closest_locale(),
            gpu_locale_id);
}

HCLIB_REGISTER_MODULE("cuda", cuda_pre_initialize, cuda_post_initialize)
