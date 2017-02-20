#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "unbalanced_bin_fan_out.h"

enum {
    RECURSE_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0
};

static Realm::Processor aggregateCpu = Realm::Processor::NO_PROC;

static void recurse(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == 2 * sizeof(int));

    int *params = (int *)args;
    const int depth = params[0];
    const int branch = params[1];

    const int depth_limit = branch * BIN_FAN_OUT_DEPTH_MULTIPLIER;

    if (depth < depth_limit) {
        int newParams[2];
        newParams[0] = depth + 1;
        newParams[1] = branch;

        aggregateCpu.spawn(RECURSE_ID, newParams, 2 * sizeof(int));
        aggregateCpu.spawn(RECURSE_ID, newParams, 2 * sizeof(int));
    }
}

static int collect_cpus(std::vector<Realm::Processor> &all_cpus) {
    std::set<Realm::Processor> all_procs;
    Realm::Machine::get_machine().get_all_processors(all_procs);

    int count_cpus = 0;
    for (std::set<Realm::Processor>::iterator i = all_procs.begin(),
            e = all_procs.end(); i != e; i++) {
        Realm::Processor curr = *i;
        if (curr.kind() == Realm::Processor::LOC_PROC) {
            all_cpus.push_back(curr);
            count_cpus++;
        }
    }

    return count_cpus;
}

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */
int main(int argc, char **argv) {
    Realm::Runtime runtime;
    std::vector<Realm::Processor> all_cpus;
    runtime.init(&argc, &argv);
    runtime.register_task(RECURSE_ID, recurse);

    int count_cpus = collect_cpus(all_cpus);
    aggregateCpu = Realm::Processor::create_group(all_cpus);
    printf("Using %d Realm threads\n", count_cpus);

    const unsigned long long start_time = hclib_current_time_ns();

    unsigned ntasks = 0;
    int i;
    for (i = 0; i < N_BRANCHES; i++) {
        int params[2];
        params[0] = 0;
        params[1] = i;

        ntasks += (1 << (i * BIN_FAN_OUT_DEPTH_MULTIPLIER));
        aggregateCpu.spawn(RECURSE_ID, params, 2 * sizeof(int));
    }

    runtime.shutdown();
    runtime.wait_for_shutdown();

    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC unbalanced_bin_fan_out %d|%d %.20f\n", N_BRANCHES,
            BIN_FAN_OUT_DEPTH_MULTIPLIER,
            (double)ntasks / ((double)(end_time - start_time) / 1000.0));

    return 0;
}
