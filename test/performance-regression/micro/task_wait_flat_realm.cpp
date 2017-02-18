#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "task_wait_flat.h"

enum {
    TASK_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0
};

static void task(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    arglen = arglen + 1;
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
    runtime.register_task(TASK_ID, task);

    int count_cpus = collect_cpus(all_cpus);
    printf("Using %d Realm threads\n", count_cpus);

    const unsigned long long start_time = hclib_current_time_ns();
    int i;
    for (i = 0; i < N_FLAT_TASK_WAITS; i++) {
        Realm::Event e = all_cpus.at(i % count_cpus).spawn(TASK_ID, NULL, 0);
        e.external_wait();
    }
    const unsigned long long end_time = hclib_current_time_ns();
    printf("METRIC task_wait_flat %d %.20f\n", N_FLAT_TASK_WAITS,
            (double)N_FLAT_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));

    runtime.shutdown();
    runtime.wait_for_shutdown();

    return 0;
}
