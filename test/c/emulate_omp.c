#include "hclib.h"
#include <assert.h>
#include <unistd.h>

static int arr[10];

void *write_arr(void *arg) {
    size_t index = (size_t)arg;
    sleep(1);
    assert(arr[index] == 0);
    arr[index] = 1;
}

void entrypoint(void *arg) {
    int i;
    assert(arg == NULL);

    hclib_start_finish();

    fprintf(stderr, "Task 0 putting on %p\n", arr + 0);
    hclib_emulate_omp_task(write_arr, (void *)((size_t)i), NULL, 0, 1, arr + 0,
            0);

    for (i = 1; i < 10; i++) {
    fprintf(stderr, "Task %d putting on %p, waiting on %p\n", i, arr + i,
            arr + (i - 1));
        hclib_emulate_omp_task(write_arr, (void *)((size_t)i), NULL, 1, 1,
                arr + (i - 1), 0, arr + i, 0);
    }

    hclib_end_finish();
}

int main(int argc, char **argv) {
    memset(arr, 0x00, 10 * sizeof(int));

    hclib_launch(entrypoint, NULL);
    return 0;
}
