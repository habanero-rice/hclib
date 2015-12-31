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

void init_contents(int *arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = 2 * i;
    }
}

void verify_contents(int *arr, int N) {
    for (int i = 0; i < N; i++) {
        assert(arr[i] == 2 * i);
    }
}

void run_copy_test(hclib::place_t *dst_pl, int *dst, hclib::place_t *src_pl,
        int *src, size_t nelems, bool dst_is_gpu, const char *test_name) {
    const size_t nbytes = nelems * sizeof(int);

    std::cerr << "Running " << std::string(test_name) << " (nbytes = " <<
        nbytes << ")" << std::endl;

    if (dst_is_gpu) {
        CHECK_CUDA(cudaMemset(dst, 0x00, nbytes));
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        memset(dst, 0x00, nbytes);
    }

    hclib::promise_t *event = hclib::async_copy(dst_pl, dst, src_pl, src, nelems,
            NULL, dst);
    void *result = hclib::promise_wait(event);
    if (result != dst) {
        fprintf(stderr, "Expected %p, got %p\n", dst, result);
        assert(0);
    }

    if (dst_is_gpu) {
        int *tmp = (int *)malloc(nelems * sizeof(int));
        assert(tmp);
        CHECK_CUDA(cudaMemcpy(tmp, dst, nbytes, cudaMemcpyDeviceToHost));
        verify_contents(tmp, nelems);
        free(tmp);
    } else {
        verify_contents(dst, nelems);
    }
}

int main(int argc, char **argv) {
    hclib::launch(&argc, argv, []() {
        hclib::place_t *my_pl = hclib::get_current_place();
        hclib::place_t *root_pl = hclib::get_root_place();

        int num_toplevel;
        hclib::place_t **toplevel = hclib::get_children_of_place(root_pl,
                &num_toplevel);
        hclib::place_t *cpu_place = NULL;
        hclib::place_t *gpu_place = NULL;

        for (int i = 0; i < num_toplevel && (cpu_place == NULL ||
                    gpu_place == NULL); i++) {
            if (toplevel[i]->type == NVGPU_PLACE) {
                std::cout << "Found a GPU place" << std::endl;
                gpu_place = toplevel[i];
            } else {
                std::cout << "Found a CPU place" << std::endl;
                cpu_place = toplevel[i];
            }
        }

        assert(cpu_place && gpu_place);

        const size_t nelems = 10;

        int *h_ptr_normal = hclib::allocate_at<int>(cpu_place, nelems, NONE);
        assert(h_ptr_normal);
        int *h_ptr_physical = hclib::allocate_at<int>(cpu_place, nelems,
                PHYSICAL);
        assert(h_ptr_physical);
        int *d_ptr = hclib::allocate_at<int>(gpu_place, nelems, NONE);
        assert(d_ptr);

        init_contents(h_ptr_normal, nelems);

        // Regular CPU -> Physical CPU
        run_copy_test(cpu_place, h_ptr_physical, cpu_place, h_ptr_normal,
                nelems, false, "regular_cpu_to_physical");

        // Physical CPU -> Regular CPU
        run_copy_test(cpu_place, h_ptr_normal, cpu_place, h_ptr_physical,
                nelems, false, "physical_cpu_to_regular");

        // Regular CPU -> GPU
        run_copy_test(gpu_place, d_ptr, cpu_place, h_ptr_normal, nelems, true,
                "regular_cpu_to_gpu");

        // GPU -> Regular CPU
        run_copy_test(cpu_place, h_ptr_normal, gpu_place, d_ptr, nelems, false,
                "gpu_to_regular_cpu");

        // Physical CPU -> GPU
        run_copy_test(gpu_place, d_ptr, cpu_place, h_ptr_physical, nelems,
                true, "physical_cpu_to_gpu");

        // GPU -> Physical CPU
        run_copy_test(cpu_place, h_ptr_physical, gpu_place, d_ptr, nelems,
                false, "gpu_to_physical_cpu");
    });
    return 0;
}
