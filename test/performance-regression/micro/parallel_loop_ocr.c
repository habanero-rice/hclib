#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "parallel_loop.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long flat_start_time = 0;
static unsigned long long recursive_start_time = 0;

#define LOOP_TILE 1000

typedef struct _loopChunk {
    u64 startInclusive;
    u64 endExclusive;
    ocrGuid_t templateGuid;
} loopChunk;

static void processLoopChunk(const u64 startInclusive, const u64 endExclusive) {
    u64 i;
    for (i = startInclusive; i < endExclusive; i++) {
        ;
    }
}

ocrGuid_t flatLoopChunk(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(paramc == 3);
    loopChunk *chunk = (loopChunk *)paramv;
    processLoopChunk(chunk->startInclusive, chunk->endExclusive);

    return NULL_GUID;
}

ocrGuid_t recursiveLoopChunk(u32 paramc, u64* paramv, u32 depc,
        ocrEdtDep_t depv[]) {
    assert(paramc == 3);
    loopChunk *chunk = (loopChunk *)paramv;
    u64 span = chunk->endExclusive - chunk->startInclusive;
    if (span <= LOOP_TILE) {
        processLoopChunk(chunk->startInclusive, chunk->endExclusive);
    } else {
        loopChunk lowChunk, highChunk;
        lowChunk.startInclusive = chunk->startInclusive;
        lowChunk.endExclusive = chunk->startInclusive + (span / 2);
        highChunk.startInclusive = lowChunk.endExclusive;
        highChunk.endExclusive = chunk->endExclusive;
        lowChunk.templateGuid = chunk->templateGuid;
        highChunk.templateGuid = chunk->templateGuid;

        ocrGuid_t task;
        ocrEdtCreate(&task, chunk->templateGuid, 3, (u64 *)&lowChunk, 0,
                NULL, EDT_PROP_NONE, NULL, NULL);
        ocrEdtCreate(&task, chunk->templateGuid, 3, (u64 *)&highChunk, 0,
                NULL, EDT_PROP_NONE, NULL, NULL);
    }

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t flatFinishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 2);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t flatTemplateGuid = params[0];

    flat_start_time = hclib_current_time_ns();

    u64 i;
    for (i = 0; i < PARALLEL_LOOP_RANGE; i += LOOP_TILE) {
        loopChunk chunk;
        chunk.startInclusive = i;
        chunk.endExclusive = i + LOOP_TILE;
        if (chunk.endExclusive > PARALLEL_LOOP_RANGE) {
            chunk.endExclusive = PARALLEL_LOOP_RANGE;
        }

        ocrGuid_t task;
        ocrEdtCreate(&task, flatTemplateGuid, 3, (u64 *)&chunk, 0, NULL,
                EDT_PROP_NONE, NULL, NULL);
    }

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t recursiveFinishEdt(u32 paramc, u64* paramv, u32 depc,
        ocrEdtDep_t depv[]) {
    assert(depc == 1);
    assert(paramc == 2);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t recurseTemplateGuid = params[1];

    const unsigned long long flat_end_time = hclib_current_time_ns();
    PRINTF("METRIC flat_parallel_iters %d %.20f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(flat_end_time -
                    flat_start_time) / 1000.0));

    recursive_start_time = hclib_current_time_ns();

    loopChunk chunk;
    chunk.startInclusive = 0;
    chunk.endExclusive = PARALLEL_LOOP_RANGE;
    chunk.templateGuid = recurseTemplateGuid;

    ocrGuid_t task;
    ocrEdtCreate(&task, recurseTemplateGuid, 3, (u64 *)&chunk, 0, NULL,
            EDT_PROP_NONE, NULL, NULL);

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);

    const unsigned long long recursive_end_time = hclib_current_time_ns();

    PRINTF("METRIC recursive_parallel_iters %d %.20f\n", PARALLEL_LOOP_RANGE,
            (double)PARALLEL_LOOP_RANGE / ((double)(recursive_end_time -
                    recursive_start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t flatTemplateGuid, recursiveTemplateGuid, flatDriverTemplateGuid,
              recursiveDriverTemplateGuid, finalTemplateGuid;
    ocrEdtTemplateCreate(&flatTemplateGuid, flatLoopChunk, 3, 0);
    ocrEdtTemplateCreate(&recursiveTemplateGuid, recursiveLoopChunk, 3, 0);
    ocrEdtTemplateCreate(&flatDriverTemplateGuid, flatFinishEdt, 2, 0);
    ocrEdtTemplateCreate(&recursiveDriverTemplateGuid, recursiveFinishEdt, 2, 1);
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t params[] = { flatTemplateGuid, recursiveTemplateGuid };

    ocrGuid_t flatTask, flatTaskEvent;
    ocrEdtCreate(&flatTask, flatDriverTemplateGuid, 2, (u64 *)params, 0, NULL,
            EDT_PROP_FINISH, NULL, &flatTaskEvent);

    ocrGuid_t recursiveTask, recursiveTaskEvent;
    ocrEdtCreate(&recursiveTask, recursiveDriverTemplateGuid, 2, (u64 *)params,
            1, &flatTaskEvent, EDT_PROP_FINISH, NULL, &recursiveTaskEvent);

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &recursiveTaskEvent,
            EDT_PROP_NONE, NULL, NULL);

    return NULL_GUID;
}

}
