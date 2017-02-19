#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "prod_cons.h"

enum {
    PRODUCER_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0,
    CONSUMER_ID
};

static Realm::Processor aggregateCpu = Realm::Processor::NO_PROC;

static void producer(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == sizeof(int *));
    int *ptr = *((int **)args);
    *ptr = 3;
}

static void consumer(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == sizeof(int *));
    int *ptr = *((int **)args);
    assert(*ptr == 3);
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
    runtime.register_task(PRODUCER_ID, producer);
    runtime.register_task(CONSUMER_ID, consumer);

    int count_cpus = collect_cpus(all_cpus);
    printf("Using %d Realm threads\n", count_cpus);
    aggregateCpu = Realm::Processor::create_group(all_cpus);

    const unsigned long long start_time = hclib_current_time_ns();

    std::set<Realm::Event> allEvents;
    int i;
    for (i = 0; i < PROD_CONS_MSGS; i++) {
        int *val = (int *)malloc(sizeof(int));
        assert(val);

        Realm::Event e = aggregateCpu.spawn(PRODUCER_ID, &val, sizeof(val));
        e = aggregateCpu.spawn(CONSUMER_ID, &val, sizeof(val), e);
        allEvents.insert(e);
    }
    Realm::Event merged = Realm::Event::merge_events(allEvents);
    merged.external_wait();

    const unsigned long long end_time = hclib_current_time_ns();

    printf("METRIC producer_consumer %d %.20f\n", PROD_CONS_MSGS,
            (double)PROD_CONS_MSGS / ((double)(end_time - start_time) /
                1000.0));

    runtime.shutdown();
    runtime.wait_for_shutdown();

    return 0;
}
