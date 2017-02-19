#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "reduce_var.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long start_time = 0;

ocrGuid_t reduceEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(paramc == 0);
    assert(depc == 2 || depc == 1);
    
    if (depc == 2) {
        assert(depv[0].ptr);
        assert(depv[1].ptr);

        int *a = (int *)depv[0].ptr;
        int *b = (int *)depv[1].ptr;

        ocrGuid_t dbGuid;
        int *outPtr;
        ocrDbCreate(&dbGuid, (void **)&outPtr, sizeof(int), DB_PROP_NONE, NULL,
                NO_ALLOC);
        *outPtr = *a + *b;

        return dbGuid;
    } else {
        return depv[0].guid;
    }
}

// Finish EDT
ocrGuid_t finishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 1);
    ocrGuid_t reduceTemplateGuid = *((ocrGuid_t *)paramv);

    start_time = hclib_current_time_ns();

    ocrGuid_t topLevel[NREDUCERS];

    int i;
    for (i = 0; i < NREDUCERS; i++) {
        ocrEventCreate(&topLevel[i], OCR_EVENT_ONCE_T, EVT_PROP_NONE);
    }

    ocrGuid_t wavefront[NREDUCERS];
    memcpy(wavefront, topLevel, NREDUCERS * sizeof(ocrGuid_t));
    int nreducers = NREDUCERS;

    while (nreducers > 1) {
        int next_nreducers = 0;

        for (i = 0; i < nreducers; i += 2) {
            int this_nreducers = nreducers - i;
            if (this_nreducers > 2) {
                this_nreducers = 2;
            }

            ocrGuid_t task, taskEvent;
            ocrEdtCreate(&task, reduceTemplateGuid, 0, NULL, this_nreducers,
                    &wavefront[i], EDT_PROP_NONE, NULL, &taskEvent);
            wavefront[next_nreducers++] = taskEvent;
        }

        nreducers = next_nreducers;
    }
    assert(nreducers == 1);

    for (i = 0; i < NREDUCERS; i++) {
        ocrGuid_t dbGuid;
        int *ptr;
        ocrDbCreate(&dbGuid, (void **)&ptr, sizeof(int), DB_PROP_NONE, NULL,
                NO_ALLOC);
        *ptr = 1;

        ocrEventSatisfy(topLevel[i], dbGuid);
    }

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC recursive_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(end_time -
                    start_time) / 1000.0));
    PRINTF("METRIC flat_reduction %d %.20f\n", NREDUCERS,
            (double)NREDUCERS / ((double)(end_time -
                    start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t reduceTemplateGuid, finishTemplateGuid, finalTemplateGuid;
    ocrEdtTemplateCreate(&reduceTemplateGuid, reduceEdt, 0, EDT_PARAM_UNK);
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 1, 0);
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    start_time = hclib_current_time_ns();

    ocrGuid_t finishTask, finishTaskEvent;
    ocrEdtCreate(&finishTask, finishTemplateGuid, 1, (u64 *)&reduceTemplateGuid,
            0, NULL, EDT_PROP_NONE, NULL, &finishTaskEvent);

    ocrGuid_t finalTask, finalTaskEvent;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &finishTaskEvent,
            EDT_PROP_NONE, NULL, NULL);

    return NULL_GUID;
}

}
