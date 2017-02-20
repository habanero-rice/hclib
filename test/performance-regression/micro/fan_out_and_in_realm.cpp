#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "fan_out_and_in.h"

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
    Realm::UserEvent userEvent = Realm::UserEvent::create_user_event();
    std::vector<Realm::Event> *all_events = new std::vector<Realm::Event>();

    unsigned i;
    for (i = 0; i < FAN_OUT_AND_IN; i++) {
        Realm::Event e = all_cpus.at(i % count_cpus).spawn(TASK_ID, NULL, 0,
                userEvent);
        all_events->push_back(e);
    }

    while (all_events->size() > 1) {
        std::vector<Realm::Event> *new_all_events =
            new std::vector<Realm::Event>();

        for (i = 0; i < all_events->size(); i += MAX_NUM_WAITS) {
            std::set<Realm::Event> merged;
            unsigned this_n_events = all_events->size() - i;
            if (this_n_events > MAX_NUM_WAITS) this_n_events = MAX_NUM_WAITS;

            unsigned j;
            for (j = i; j < i + this_n_events; j++) {
                merged.insert(all_events->at(j));
            }

            Realm::Event mergedEvent = Realm::Event::merge_events(merged);
            Realm::Event finalEvent = all_cpus
                .at(new_all_events->size() % count_cpus)
                .spawn(TASK_ID, NULL, 0, mergedEvent);
            new_all_events->push_back(finalEvent);
        }
        all_events = new_all_events;
    }
    assert(all_events->size() == 1);

    userEvent.trigger();
    all_events->at(0).external_wait();
    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC fan_out_and_in %d %.20f\n", FAN_OUT_AND_IN,
            (double)FAN_OUT_AND_IN / ((double)(end_time - start_time) /
                1000.0));

    runtime.shutdown();
    runtime.wait_for_shutdown();

    return 0;
}
