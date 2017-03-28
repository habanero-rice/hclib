#include "hclib_cpp.h"
#include "hclib_sos.h"
#include "hclib_system.h"

#include <unistd.h>
#include <iostream>

int shared_var_0 = 0;
int shared_var_1 = 0;

int main(int argc, char **argv) {
    const char *deps[] = { "system", "sos" };
    hclib::launch(deps, 2, [] {
        const int pe = hclib::shmem_my_pe();
        std::cout << "Hello world from rank " << pe << std::endl;

        volatile int **wait_vars = (volatile int **)malloc(2 * sizeof(int *));
        wait_vars[0] = &shared_var_0;
        wait_vars[1] = &shared_var_1;

        int *cmp_values = (int *)malloc(2 * sizeof(int));
        cmp_values[0] = pe + 1;
        cmp_values[1] = pe + 1;

        hclib::shmem_barrier_all();

        if (pe < hclib::shmem_n_pes() - 1) {
            const unsigned long long start_wait = hclib_current_time_ns();
            hclib::shmem_int_async_when_any(wait_vars, SHMEM_CMP_EQ,
                cmp_values, 2, [start_wait, pe] {
                    const unsigned long long end_wait = hclib_current_time_ns();
                    assert((shared_var_0 == pe + 1 ||
                        shared_var_1 == pe + 1) &&
                        shared_var_0 != shared_var_1);

                    std::cerr << "PE " << pe << " waited for " <<
                        ((end_wait - start_wait) / 1000000) << " ms" << std::endl;

                    sleep(1);

                    if (pe > 0) {
                        int pe_no = pe;
                        if (pe_no % 2 == 0) {
                            hclib::shmem_int_put(&shared_var_0, &pe_no, 1,
                                    pe - 1);
                        } else {
                            hclib::shmem_int_put(&shared_var_1, &pe_no, 1,
                                    pe - 1);
                        }
                    }
                });
        } else {
            int pe_no = pe;
            if (pe_no % 2 == 0) {
                hclib::shmem_int_put(&shared_var_0, &pe_no, 1,
                        pe - 1);
            } else {
                hclib::shmem_int_put(&shared_var_1, &pe_no, 1,
                        pe - 1);
            }
        }

        // hclib::shmem_barrier_all();
    });
    return 0;
}
