#include "hclib_cpp.h"
#include "hclib_cuda.h"
#include "hclib_system.h"

#include <iostream>

int main(int argc, char **argv) {
    hclib::launch([] {
        int gpu_id = hclib::get_gpu_locale_id();
        hclib::locale_t *gpu_locale = hclib::get_closest_gpu_locale();
        std::cout << "Hello world, gpu locale id = " << gpu_id <<
            ", closest gpu = " << gpu_locale << std::endl;
        assert(gpu_locale);

        hclib_future_t *fut = hclib::allocate_at(1024, gpu_locale);
        void *alloc = hclib_future_wait(fut);
        assert(alloc);
    });
    return 0;
}
