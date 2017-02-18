#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "task_wait_recursive.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

typedef struct _finishEdtParams {
    u64 depth;
    ocrGuid_t finishTemplateGuid;
} finishEdtParams;

static unsigned long long start_time = 0;

// Finish EDT
ocrGuid_t finishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0 || depc == 1);
    assert(paramc == 2);
    finishEdtParams *params = (finishEdtParams *)paramv;
    u64 depth = params->depth;
    ocrGuid_t finishTemplateGuid = params->finishTemplateGuid;

    if (depth < N_RECURSIVE_TASK_WAITS) {
        ocrGuid_t task;
        params->depth += 1;
        ocrEdtCreate(&task, finishTemplateGuid, 2, paramv, 0, NULL,
                EDT_PROP_FINISH, NULL, NULL);
    }

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC task_wait_recursive %d %.20f\n", N_RECURSIVE_TASK_WAITS,
            (double)N_RECURSIVE_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs
    assert(sizeof(finishEdtParams) == 2 * sizeof(u64));

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t finishTemplateGuid;
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 2, EDT_PARAM_UNK);

    finishEdtParams params = { 0, finishTemplateGuid };

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

    start_time = hclib_current_time_ns();
    ocrGuid_t topLevelTask, topLevelEvent;
    ocrEdtCreate(&topLevelTask, finishTemplateGuid, 2, (u64 *)&params, 1,
            &initEvent, EDT_PROP_FINISH, NULL, &topLevelEvent);

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &topLevelEvent,
            EDT_PROP_NONE, NULL, NULL);

    ocrEventSatisfy(initEvent, NULL_GUID);

    return NULL_GUID;
}

}
