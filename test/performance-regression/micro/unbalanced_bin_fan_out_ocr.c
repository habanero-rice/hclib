#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "unbalanced_bin_fan_out.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long start_time = 0;
static unsigned ntasks = 0;

typedef struct _paramStruct {
    u64 depth;
    u64 branch;
    ocrGuid_t recurseTemplateGuid;
} paramStruct;

ocrGuid_t recurseEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(paramc == 3);
    assert(depc == 0);
    paramStruct *params = (paramStruct *)paramv;

    const unsigned depth_limit = params->branch * BIN_FAN_OUT_DEPTH_MULTIPLIER;

    if (params->depth < depth_limit) {
        ocrGuid_t task;
        params->depth += 1;
        ocrEdtCreate(&task, params->recurseTemplateGuid, 3, paramv, 0, NULL,
                EDT_PROP_NONE, NULL, NULL);
        ocrEdtCreate(&task, params->recurseTemplateGuid, 3, paramv, 0, NULL,
                EDT_PROP_NONE, NULL, NULL);
    }

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t finishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 1);
    ocrGuid_t recurseTemplateGuid = *((ocrGuid_t *)paramv);

    start_time = hclib_current_time_ns();

    int i;
    for (i = 0; i < N_BRANCHES; i++) {
        paramStruct params;
        params.recurseTemplateGuid = recurseTemplateGuid;
        params.depth = 0;
        params.branch = i;

        ntasks += (1 << (i * BIN_FAN_OUT_DEPTH_MULTIPLIER));

        ocrGuid_t task;
        ocrEdtCreate(&task, recurseTemplateGuid, 3, (u64 *)&params, 0, NULL,
                EDT_PROP_NONE, NULL, NULL);
    }

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC unbalanced_bin_fan_out %d|%d %.20f\n", N_BRANCHES,
            BIN_FAN_OUT_DEPTH_MULTIPLIER,
            (double)ntasks / ((double)(end_time - start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t recurseTemplateGuid;
    ocrEdtTemplateCreate(&recurseTemplateGuid, recurseEdt, 3, 0);

    ocrGuid_t finishTemplateGuid;
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 1, 0);

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t finishTask, finishTaskEvent;
    ocrEdtCreate(&finishTask, finishTemplateGuid, 1,
            (u64 *)&recurseTemplateGuid, 0, NULL, EDT_PROP_FINISH, NULL,
            &finishTaskEvent);

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &finishTaskEvent,
            EDT_PROP_NONE, NULL, NULL);

    return NULL_GUID;
}

}
