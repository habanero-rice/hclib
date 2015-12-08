#include <stdlib.h>
#include <stdio.h>
#include <hclib.h>
#include <unistd.h>
#include <stdbool.h>

#define FINISH GEN_FINISH_SCOPE(MACRO_CONCAT(_hcGenVar_, __COUNTER__))
#define GEN_FINISH_SCOPE(V) for (int V=(start_finish(), 1); V; end_finish(), --V)
#define MACRO_CONCAT(a, b) DO_MACRO_CONCAT(a, b)
#define DO_MACRO_CONCAT(a, b) a ## b

#if 1
#define DELAY(t) usleep(t*100000L)
#else
#define DELAY(t) /* no delay */
#endif

extern int rt_get_worker_id(void);

static inline void echo_worker(const char *taskName) {
    printf("Task %s run by worker %d\n", taskName, rt_get_worker_id());
}

struct ddf_st **ddf_list;
struct ddf_st *ddf;
int data;

void taskSleep(void *args) {
    echo_worker(args ? "Sleep A" : "Sleep B");
    DELAY(5);
}

void taskA(void *args) {
    echo_worker("A");
    FINISH {
        async(taskSleep, taskA, NO_DDF, NO_PHASER, NO_PROP);
        DELAY(2);
    }
    printf("%p <- %p\n", ddf, &data);
    ddf_put(ddf, &data);
}

void taskB(void *args) {
    echo_worker("B");
    FINISH {
        async(taskSleep, NULL, ddf_list, NO_PHASER, NO_PROP);
    }
}

void taskC(void *args) {
    echo_worker("C");
    DELAY(1);
    async(taskB, NULL, NO_DDF, NO_PHASER, NO_PROP);
    DELAY(5);
}

void taskMain(void *args) {
    ddf_list = ddf_create_n(1, true);
    ddf = ddf_create();
    ddf_list[0] = ddf;

    async(&taskA, NULL, NO_DDF, NO_PHASER, NO_PROP);
    async(&taskC, NULL, NO_DDF, NO_PHASER, NO_PROP);
    DELAY(10);
}

int main(int argc, char ** argv) {
    hclib_launch(&argc, argv, taskMain, NULL);
    printf("Done\n");
}
