#include "hclib.h"
#include <assert.h>
#include <unistd.h>

#define ARR_DIM 10

static int arr[ARR_DIM];

void *write_arr(void *arg) {
    size_t index = (size_t)arg;
    sleep(1);

    fprintf(stderr, "Running task w/ index = %d\n", (int)index);

    const int curr_val = arr[index];
    if (curr_val != 0) {
        fprintf(stderr, "Expected 0 at index %d but saw %d\n", (int)index,
                curr_val);
        assert(false);
    }

    arr[index] = 1;
}

void entrypoint(void *arg) {
    int i = 0;
    assert(arg == NULL);

    hclib_start_finish();

    fprintf(stderr, "Task 0 putting on %p\n", arr + 0);
    hclib_emulate_omp_task(write_arr, (void *)((size_t)i), NULL, 0, 1, arr + 0,
            0);

    for (i = 1; i < ARR_DIM; i++) {
    fprintf(stderr, "Task %d putting on %p, waiting on %p\n", i, arr + i,
            arr + (i - 1));
        hclib_emulate_omp_task(write_arr, (void *)((size_t)i), NULL, 1, 1,
                arr + (i - 1), 0, arr + i, 0);
    }

    hclib_end_finish();
}

int main(int argc, char **argv) {
    memset(arr, 0x00, ARR_DIM * sizeof(int));

    char const *deps[] = { "system" };
    hclib_launch(entrypoint, NULL, deps, 1);
    return 0;
}
