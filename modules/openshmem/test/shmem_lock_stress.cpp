#include "hclib_cpp.h"
#include "hclib_openshmem.h"
#include "hclib_system.h"

#include <iostream>

unsigned long long get_clock_gettime() {
    struct timespec t ={0,0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    unsigned long long s = 1000000000ULL * (unsigned long long)t.tv_sec;
    return (unsigned long long)t.tv_nsec + s;
}

long lock = 0L;

int main(int argc, char **argv) {
    const char *deps[] = { "system", "openshmem" };
    hclib::launch(deps, 2, [] {
        const int pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << pe << std::endl;

        hclib::finish([pe] {
            size_t nlocks = 0;
            const unsigned long long start_time = get_clock_gettime();
            while (get_clock_gettime() - start_time < 20000000ULL) { // 20 ms
                hclib::async([] {
                    hclib::shmem_set_lock(&lock);
                    hclib::shmem_clear_lock(&lock);
                });
                nlocks++;
                // fprintf(stderr, "%llu\n", get_clock_gettime() - start_time);
            }
            printf("PE %d issued %lu locks\n", pe, nlocks);
        });

        hclib::shmem_barrier_all();
    });

    return 0;
}
