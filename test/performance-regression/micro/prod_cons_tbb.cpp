#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"

#include <stdio.h>
#include "prod_cons.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    int nmessages = 0;
    tbb::flow::graph g;

    const unsigned long long start_time = hclib_current_time_ns();
    tbb::flow::source_node<int> source(g,
            [&] (int &next_msg) -> bool {
                next_msg = 0;
                nmessages++;
                return nmessages < PROD_CONS_MSGS;
            });

    tbb::flow::function_node<int, int> dest(g, tbb::flow::unlimited,
            [&] (int msg) -> int {
                return ++msg;
            });

    tbb::flow::buffer_node<int> outBuf(g);

    make_edge(dest, outBuf);
    make_edge(source, dest);

    g.wait_for_all();
    const unsigned long long end_time = hclib_current_time_ns();
    assert(nmessages == PROD_CONS_MSGS);
    printf("METRIC producer_consumer %d %.20f\n", PROD_CONS_MSGS,
            (double)PROD_CONS_MSGS / ((double)(end_time - start_time) /
                1000.0));

    return 0;
}
