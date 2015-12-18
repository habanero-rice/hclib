#include <iostream>
#include <assert.h>
#include <cuda_runtime_api.h>

#include "hclib_cpp.h"

#define CHECK_CUDA(call) { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

class test_functor {
    private:
        int *arr;
    public:
        test_functor(int *set_arr) : arr(set_arr) { }
        __host__ __device__ void call(int idx) {
            arr[idx] = idx;
        }
};

int main(int argc, char **argv) {
    hclib::launch(&argc, argv, []() {
        hclib::place_t *root_pl = hclib::get_root_place();

        int num_toplevel;
        hclib::place_t **toplevel = hclib::get_children_of_place(root_pl,
                &num_toplevel);
        hclib::place_t *gpu_place = NULL;
        hclib::place_t *cpu_place = NULL;
        for (int i = 0; i < num_toplevel && (cpu_place == NULL ||
                    gpu_place == NULL); i++) {
            if (toplevel[i]->type == NVGPU_PLACE) {
                gpu_place = toplevel[i];
            } else {
                cpu_place = toplevel[i];
            }
        }
        assert(gpu_place && cpu_place);

        const int N = 1024;
        int *arr = (int *)hclib::allocate_at(cpu_place, N * sizeof(int), 0);
        assert(arr);

        hclib_ddf_t *cpu_memset_event = hclib::async_memset(cpu_place, arr, 0,
                N * sizeof(int), arr);
        hclib::ddf_wait(cpu_memset_event);

        loop_domain_t loop = {0, N, 1, 33};
        test_functor kernel(arr);
        hclib::ddf_t *cpu_kernel_event = hclib::forasync1D_future_(
                (loop_domain_t *)&loop, kernel, FORASYNC_MODE_FLAT, cpu_place);

        hclib::ddf_wait(cpu_kernel_event);

        int *d_arr = (int *)hclib::allocate_at(gpu_place, N * sizeof(int), 0);
        assert(d_arr);

        hclib::ddf_t *gpu_memset_event = hclib::async_memset(gpu_place, d_arr,
                0, N * sizeof(int), d_arr);
        hclib::ddf_wait(gpu_memset_event);

        hclib::ddf_t *gpu_kernel_event = hclib::forasync1D_future_(
                (loop_domain_t *)&loop, kernel, FORASYNC_MODE_FLAT, gpu_place);
        hclib::ddf_wait(gpu_kernel_event);

        for (int i = 0; i < N; i++) {
            assert(arr[i] == i);
        }

        printf("Passed!\n");
    });
    return 0;
}
