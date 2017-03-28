#include "hclib_cpp.h"
#include "hclib_cuda.h"
#include "hclib_system.h"

#include <iostream>

static void helper() {
    hclib::locale_t *gpu_locale = hclib::get_closest_gpu_locale();
    hclib::locale_t *cpu_locale = hclib::get_closest_cpu_locale();

    const int N = 256;

    hclib::future_t<void*> *allocation = hclib::allocate_at(N * sizeof(int), gpu_locale);
    int *dArr = (int *)allocation->wait_and_get();

    allocation = hclib::allocate_at(N * sizeof(int), cpu_locale);
    int *hArr = (int *)allocation->wait_and_get();

    hclib::future_t<void> *kernel_future = hclib::async_simd([=] __host__ __device__ (int id, int width) {
        dArr[id] = width;
    }, gpu_locale);

    allocation = hclib::async_copy_await(cpu_locale, hArr, gpu_locale, dArr,
            N * sizeof(int), kernel_future);
    allocation->wait();

    int width = hArr[0];
    printf("Width = %d\n", width);

    for (int i = 0; i < width; i++) {
        assert(hArr[i] == width);
    }
}

int main(int argc, char **argv) {
    const char *deps[] = { "system", "cuda" };

    hclib::launch(deps, 2, [&] {
        helper();
    });

    return 0;
}
