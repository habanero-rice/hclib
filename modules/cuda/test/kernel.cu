#include "hclib_cpp.h"
#include "hclib_cuda.h"
#include "hclib_system.h"

#include <iostream>

class incr {
    private:
        int *arr;

    public:
        incr(int *set_arr) : arr(set_arr) { }

        __host__ __device__ void operator()(int idx) {
            arr[idx] = arr[idx] + arr[idx];
        }
};

int main(int argc, char **argv) {
    const char *deps[] = { "system", "cuda" };
    hclib::launch(deps, 2, [] {
        const int N = 1024;
        int gpu_id = hclib::get_gpu_locale_id();
        int num_gpus = hclib::get_num_gpu_locales();
        assert(num_gpus > 0);

        hclib::locale_t *cpu_locale = hclib::get_closest_cpu_locale();
        hclib::locale_t *gpu_locale = hclib::get_closest_gpu_locale();

        std::cout << "Hello world, gpu locale id = " << gpu_id <<
            ", closest gpu = " << gpu_locale << " (" <<
            hclib::get_gpu_name(gpu_locale) << ")" << std::endl;

        hclib::finish([N, cpu_locale, gpu_locale] {
            hclib::future_t *cpu_fut = hclib::allocate_at(N * sizeof(int),
                    cpu_locale);
            hclib::future_t *gpu_fut = hclib::allocate_at(N * sizeof(int),
                    gpu_locale);
            hclib::future_t *init_fut = hclib::async_future_await_at([N, cpu_fut] {
                        int *host_alloc = (int *)cpu_fut->get();
                        for (int i = 0; i < N; i++) {
                            host_alloc[i] = i;
                        }
                    }, cpu_locale, cpu_fut);

            hclib::async_await_at([cpu_fut, gpu_fut, init_fut, N, cpu_locale, gpu_locale] {
                        int *gpu_mem = (int *)gpu_fut->get();
                        int *cpu_mem = (int *)cpu_fut->get();

                        hclib::future_t *copy_to_fut = hclib::async_copy(
                                gpu_locale, gpu_mem, cpu_locale, cpu_mem,
                                N * sizeof(int));
                        hclib::future_t *kernel_fut = hclib::forasync_cuda(N,
                                incr(gpu_mem), gpu_locale, copy_to_fut);
                        hclib::future_t *copy_back_fut = hclib::async_copy_await(
                                cpu_locale, cpu_mem, gpu_locale, gpu_mem,
                                N * sizeof(int), kernel_fut);
                        hclib::async_await_at([cpu_mem, N] {
                                    for (int i = 0; i < N; i++) {
                                        assert(cpu_mem[i] == 2 * i);
                                    }
                                }, cpu_locale, copy_back_fut);
                    }, cpu_locale, gpu_fut, init_fut);
        });
    });
}
