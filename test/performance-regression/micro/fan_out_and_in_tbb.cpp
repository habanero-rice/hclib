#include "hclib.h"

#include "tbb/task_scheduler_init.h"
#include "tbb/flow_graph.h"

#include <stdio.h>
#include "fan_out_and_in.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    // Code below relies on this for TBB join nodes

    int nthreads = tbb::task_scheduler_init::default_num_threads();
    printf("Using %d TBB threads\n", nthreads);

    tbb::flow::graph g;

    std::vector<tbb::flow::function_node<int, int> *> *fan_out_nodes =
        new std::vector<tbb::flow::function_node<int, int> *>();

    const unsigned long long start_time = hclib_current_time_ns();
    tbb::flow::broadcast_node<int> *root = new tbb::flow::broadcast_node<int>(g);
    unsigned i;
    for (i = 0; i < FAN_OUT_AND_IN; i++) {
        tbb::flow::function_node<int, int> *new_node =
            new tbb::flow::function_node<int, int>(g, tbb::flow::unlimited,
                    [=](int in) -> int {
                        int incr = in + 1;
                        return incr;
                    });
        new_node->register_predecessor(*root);
        fan_out_nodes->push_back(new_node);
    }

    std::vector<tbb::flow::function_node<tbb::flow::tuple<int,int>, int> *> *join_nodes =
        new std::vector<tbb::flow::function_node<tbb::flow::tuple<int,int>, int> *>();
    for (i = 0; i < fan_out_nodes->size(); i += 2) {
        int this_n_futures = fan_out_nodes->size() - i;
        if (this_n_futures > 2) this_n_futures = 2;

        tbb::flow::join_node<tbb::flow::tuple<int,int> > *join =
            new tbb::flow::join_node<tbb::flow::tuple<int,int> >(g);
     
        tbb::flow::make_edge(*fan_out_nodes->at(i),
                tbb::flow::input_port<0>(*join));
        if (this_n_futures > 1) {
            tbb::flow::make_edge(*fan_out_nodes->at(i + 1),
                    tbb::flow::input_port<1>(*join));
        }

        tbb::flow::function_node<tbb::flow::tuple<int, int>, int> *join_filter =
            new tbb::flow::function_node<tbb::flow::tuple<int, int>, int>(g,
                    tbb::flow::unlimited, [=](tbb::flow::tuple<int, int> in) -> int {
                    return std::get<0>(in);
                });
        join_filter->register_predecessor(*join);

        join_nodes->push_back(join_filter);
    }

    while (join_nodes->size() > 1) {
        std::vector<tbb::flow::function_node<tbb::flow::tuple<int,int>, int> *> *next_join_nodes =
            new std::vector<tbb::flow::function_node<tbb::flow::tuple<int,int>, int> *>();

        for (i = 0; i < join_nodes->size(); i += 2) {
            int this_n_futures = join_nodes->size() - i;
            if (this_n_futures > 2) this_n_futures = 2;

            tbb::flow::join_node<tbb::flow::tuple<int,int> > *join =
                new tbb::flow::join_node<tbb::flow::tuple<int,int> >(g);
         
            tbb::flow::make_edge(*join_nodes->at(i),
                    tbb::flow::input_port<0>(*join));
            if (this_n_futures > 1) {
                tbb::flow::make_edge(*join_nodes->at(i + 1),
                        tbb::flow::input_port<1>(*join));
            }

            tbb::flow::function_node<tbb::flow::tuple<int, int>, int> *join_filter =
                new tbb::flow::function_node<tbb::flow::tuple<int, int>, int>(g,
                        tbb::flow::unlimited, [=](tbb::flow::tuple<int, int> in) -> int {
                        return std::get<0>(in);
                    });
            join_filter->register_predecessor(*join);

            join_nodes->push_back(join_filter);
        }

        join_nodes = next_join_nodes;
    }

    bool success = root->try_put(0);
    assert(success);
    g.wait_for_all();
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC fan_out_and_in %d %.20f\n", FAN_OUT_AND_IN,
            (double)FAN_OUT_AND_IN / ((double)(end_time - start_time) /
                1000.0));

    return 0;
}
