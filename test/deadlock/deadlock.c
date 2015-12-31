#include <stdlib.h>
#include <stdio.h>
#include <hclib.h>
#include <unistd.h>
#include <stdbool.h>

#define FINISH GEN_FINISH_SCOPE(MACRO_CONCAT(_hcGenVar_, __COUNTER__))
#define GEN_FINISH_SCOPE(V) for (int V=(hclib_start_finish(), 1); V; hclib_end_finish(), --V)
#define MACRO_CONCAT(a, b) DO_MACRO_CONCAT(a, b)
#define DO_MACRO_CONCAT(a, b) a ## b

#if 1
#define DELAY(t) usleep(t*100000L)
#else
#define DELAY(t) /* no delay */
#endif

extern int get_current_worker();

static inline void echo_worker(const char *taskName) {
    printf("Task %s run by worker %d\n", taskName, get_current_worker());
}

hclib_promise_t **promise_list;
hclib_promise_t *promise;
int data;

void taskSleep(void *args) {
    echo_worker(args ? "Sleep A" : "Sleep B");
    DELAY(5);
}

void taskA(void *args) {
    echo_worker("A");
    FINISH {
        hclib_async(taskSleep, taskA, NO_PROMISE, NO_PHASER, NO_PROP);
        DELAY(2);
    }
    printf("%p <- %p\n", promise, &data);
    hclib_promise_put(promise, &data);
}

void taskB(void *args) {
    echo_worker("B");
    FINISH {
        hclib_async(taskSleep, NULL, promise_list, NO_PHASER, NO_PROP);
    }
}

void taskC(void *args) {
    echo_worker("C");
    DELAY(1);
    hclib_async(taskB, NULL, NO_PROMISE, NO_PHASER, NO_PROP);
    DELAY(5);
}

void taskMain(void *args) {
    promise_list = hclib_promise_create_n(1, true);
    promise = hclib_promise_create();
    promise_list[0] = promise;

    hclib_async(&taskA, NULL, NO_PROMISE, NO_PHASER, NO_PROP);
    hclib_async(&taskC, NULL, NO_PROMISE, NO_PHASER, NO_PROP);
    DELAY(10);
}

int main(int argc, char ** argv) {
    hclib_launch(&argc, argv, taskMain, NULL);
    printf("Done\n");
}
