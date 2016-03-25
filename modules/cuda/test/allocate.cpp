#include "hclib_cpp.h"
#include "hclib_cuda.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        int gpu_id = hclib::get_gpu_locale_id();
        hclib::locale_t *gpu_locale = hclib::get_closest_gpu_locale();
        std::cout << "Hello world, gpu locale id = " << gpu_id <<
            ", closest gpu = " << gpu_locale << " (" <<
            hclib::get_gpu_name(gpu_locale) << ")" << std::endl;
        assert(gpu_locale);

        int ngpus;
        hclib::locale_t **gpu_locales = hclib::get_gpu_locales(&ngpus);
        std::cout << "Got " << ngpus << " GPU locale(s)" << std::endl;

        hclib_future_t *fut = hclib::allocate_at(1024, gpu_locale);
        void *alloc = hclib_future_wait(fut);
        assert(alloc);

        fut = hclib::async_copy(gpu_locale, alloc, cpu_locale,
                host_alloc, 1024);
        hclib_future_wait(fut);
        //TODO verify and add the cpu_locale getter
    });
    return 0;
}
