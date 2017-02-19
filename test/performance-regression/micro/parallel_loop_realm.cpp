#include "hclib.h"

#include "realm/realm.h"
#include <stdio.h>
#include "parallel_loop.h"

#define LOOP_TILE 1000

typedef struct _loopChunk {
    size_t startInclusive;
    size_t endExclusive;
} loopChunk;

static Realm::Processor aggregateCpu = Realm::Processor::NO_PROC;

enum {
    FLAT_TASK_ID = Realm::Processor::TASK_ID_FIRST_AVAILABLE + 0,
    RECURSIVE_TASK_ID
};

static void processLoopChunk(const size_t startInclusive,
        const size_t endExclusive) {
    size_t i;
    for (i = startInclusive; i < endExclusive; i++) {
        ;
    }
}

static void flatTask(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == sizeof(loopChunk));
    loopChunk *chunk = (loopChunk *)args;
    processLoopChunk(chunk->startInclusive, chunk->endExclusive);
}

static void recursiveTask(const void *args, size_t arglen, const void *userdata,
        size_t userlen, Realm::Processor p) {
    assert(arglen == sizeof(loopChunk));
    loopChunk *chunk = (loopChunk *)args;
    size_t span = chunk->endExclusive - chunk->startInclusive;

    if (span <= LOOP_TILE) {
        processLoopChunk(chunk->startInclusive, chunk->endExclusive);
    } else {
        loopChunk lowChunk, highChunk;
        lowChunk.startInclusive = chunk->startInclusive;
        lowChunk.endExclusive = chunk->startInclusive + (span / 2);
        highChunk.startInclusive = lowChunk.endExclusive;
        highChunk.endExclusive = chunk->endExclusive;

        Realm::Event e1 = aggregateCpu.spawn(RECURSIVE_TASK_ID, &lowChunk,
                sizeof(lowChunk));
        Realm::Event e2 = aggregateCpu.spawn(RECURSIVE_TASK_ID, &highChunk,
                sizeof(highChunk));
        Realm::Event merged = Realm::Event::merge_events(e1, e2);
        merged.wait();
    }
}

static void recursiveTaskWrapper(const void *args, size_t arglen,
        const void *userdata, size_t userlen, Realm::Processor p) {
    loopChunk chunk;
    chunk.startInclusive = 0;
    chunk.endExclusive = PARALLEL_LOOP_RANGE;

    Realm::Event e = aggregateCpu.spawn(RECURSIVE_TASK_ID, &chunk,
            sizeof(chunk));
    e.wait();
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
    runtime.register_task(FLAT_TASK_ID, flatTask);
    runtime.register_task(RECURSIVE_TASK_ID, recursiveTask);

    int count_cpus = collect_cpus(all_cpus);
    printf("Using %d Realm threads\n", count_cpus);
    aggregateCpu = Realm::Processor::create_group(all_cpus);

    const unsigned long long flat_start_time = hclib_current_time_ns();
    std::set<Realm::Event> flatEvents;
    size_t i;
    for (i = 0; i < PARALLEL_LOOP_RANGE; i += LOOP_TILE) {
        loopChunk chunk;
        chunk.startInclusive = i;
        chunk.endExclusive = i + LOOP_TILE;
        if (chunk.endExclusive > PARALLEL_LOOP_RANGE) {
            chunk.endExclusive = PARALLEL_LOOP_RANGE;
        }
        flatEvents.insert(aggregateCpu.spawn(FLAT_TASK_ID, &chunk, sizeof(chunk)));
    }
    Realm::Event flatMerged = Realm::Event::merge_events(flatEvents);
    flatMerged.external_wait();
    const unsigned long long flat_end_time = hclib_current_time_ns();
    printf("METRIC flat_parallel_iters %d %.20f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(flat_end_time -
                    flat_start_time) / 1000.0));

    const unsigned long long recursive_start_time = hclib_current_time_ns();
    loopChunk recursiveChunk;
    recursiveChunk.startInclusive = 0;
    recursiveChunk.endExclusive = PARALLEL_LOOP_RANGE;
    Realm::Event e = aggregateCpu.spawn(RECURSIVE_TASK_ID, &recursiveChunk,
            sizeof(recursiveChunk));
    e.external_wait();
    const unsigned long long recursive_end_time = hclib_current_time_ns();
    printf("METRIC recursive_parallel_iters %d %.20f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(recursive_end_time -
                    recursive_start_time) / 1000.0));

    runtime.shutdown();
    runtime.wait_for_shutdown();

    return 0;
}
