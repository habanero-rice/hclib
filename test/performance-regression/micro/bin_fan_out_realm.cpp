#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "bin_fan_out.h"

enum {
    RECURSE_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0
};

static Realm::Processor aggregateCpu = Realm::Processor::NO_PROC;

static void recurse(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == sizeof(int));
    int depth = *((int *)args);

    if (depth < BIN_FAN_OUT_DEPTH) {
        const int newDepth = depth + 1;
        aggregateCpu.spawn(RECURSE_ID, &newDepth, sizeof(newDepth));
        aggregateCpu.spawn(RECURSE_ID, &newDepth, sizeof(newDepth));
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

    const int depth = 0;
    aggregateCpu.spawn(RECURSE_ID, &depth, sizeof(depth));

    runtime.shutdown();
    runtime.wait_for_shutdown();

    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC bin_fan_out %d %.20f\n", BIN_FAN_OUT_DEPTH,
            (double)(1 << BIN_FAN_OUT_DEPTH) /
            ((double)(end_time - start_time) / 1000.0));

    return 0;
}
