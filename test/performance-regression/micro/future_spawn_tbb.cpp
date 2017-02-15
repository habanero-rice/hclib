#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"

#include <stdio.h>
#include "future_spawn.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    int nlaunched = 0;
    tbb::flow::graph g;
    tbb::flow::function_node<int, int> *first = NULL;
    tbb::flow::function_node<int, int> *prev = NULL;

    const unsigned long long spawn_start_time = hclib_current_time_ms();
    do {
        tbb::flow::function_node<int, int> *new_node =
            new tbb::flow::function_node<int, int>(g, tbb::flow::unlimited,
                    [=]( int in ) -> int {
                        int incr = in + 1;
                        return incr;
                    });

        if (first == NULL) {
            first = new_node;
        }

        if (prev) {
            new_node->register_predecessor(*prev);
        }

        prev = new_node;
        nlaunched++;
    } while (nlaunched < NFUTURES);
    bool success = first->try_put(0);
    assert(success);
    const unsigned long long spawn_end_time = hclib_current_time_ns();
    printf("METRIC future_create %d %f\n", NFUTURES,
            (double)NFUTURES / ((double)(spawn_end_time - spawn_start_time) /
                1000.0));
    g.wait_for_all();

    nlaunched = 0;
    first = NULL;
    prev = NULL;
    const unsigned long long schedule_start_time = hclib_current_time_ms();
    do {
        tbb::flow::function_node<int, int> *new_node =
            new tbb::flow::function_node<int, int>(g, tbb::flow::unlimited,
                    [=]( int in ) -> int {
                        int incr = in + 1;
                        return incr;
                    });

        if (first == NULL) {
            first = new_node;
        }

        if (prev) {
            new_node->register_predecessor(*prev);
        }

        prev = new_node;
        nlaunched++;
    } while (nlaunched < NFUTURES);
    success = first->try_put(0);
    assert(success);
    const unsigned long long schedule_end_time = hclib_current_time_ns();
    printf("METRIC future_run %d %f\n", NFUTURES,
            (double)NFUTURES / ((double)(schedule_end_time -
                    schedule_start_time) / 1000.0));

    return 0;
}
