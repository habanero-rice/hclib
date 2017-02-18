#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "future_spawn.h"

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

    {
        int nlaunched = 0;
        const unsigned long long spawn_start_time = hclib_current_time_ns();
        Realm::Event prev = Realm::Event::NO_EVENT;
        do {
            prev = all_cpus.at(nlaunched % count_cpus).spawn(TASK_ID,
                    NULL, 0, prev);
            nlaunched++;
        } while (nlaunched < NFUTURES);
        const unsigned long long spawn_end_time = hclib_current_time_ns();
        printf("METRIC future_create %d %.20f\n", NFUTURES,
                (double)NFUTURES / ((double)(spawn_end_time -
                        spawn_start_time) / 1000.0));

        prev.external_wait();
    }

    {
        int nlaunched = 0;
        const unsigned long long schedule_start_time = hclib_current_time_ns();
        Realm::Event prev = Realm::Event::NO_EVENT;
        do {
            prev = all_cpus.at(nlaunched % count_cpus).spawn(TASK_ID,
                    NULL, 0, prev);
            nlaunched++;
        } while (nlaunched < NFUTURES);
        prev.external_wait();
        const unsigned long long schedule_end_time = hclib_current_time_ns();
        printf("METRIC future_run %d %.20f\n", NFUTURES,
                (double)NFUTURES / ((double)(schedule_end_time -
                        schedule_start_time) / 1000.0));
    }

    runtime.shutdown();
    runtime.wait_for_shutdown();

    return 0;
}
