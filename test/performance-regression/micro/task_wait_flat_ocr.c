#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "task_wait_flat.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long start_time = 0;

ocrGuid_t taskEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    int incr = 0;
    incr = incr + 1;
    ocrGuid_t ret = { 0 };
    return ret;
}

// Finish EDT
ocrGuid_t finishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1);
    assert(paramc == 1);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t taskTemplateGuid = params[0];

    ocrGuid_t task;
    ocrEdtCreate(&task, taskTemplateGuid, 0, NULL, 0, NULL, EDT_PROP_NONE, NULL,
            NULL);

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC task_wait_flat %d %.20f\n", N_FLAT_TASK_WAITS,
            (double)N_FLAT_TASK_WAITS / ((double)(end_time -
                    start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t taskTemplateGuid;
    ocrEdtTemplateCreate(&taskTemplateGuid, taskEdt, 0, 0);

    ocrGuid_t params[] = { taskTemplateGuid };

    ocrGuid_t finishTemplateGuid;
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 1, 1);

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);
    ocrGuid_t prevEvent = initEvent;

    start_time = hclib_current_time_ns();
    int i;
    for (i = 0; i < N_FLAT_TASK_WAITS; i++) {
        ocrGuid_t finish, finishEvent;
        ocrEdtCreate(&finish, finishTemplateGuid, 1, (u64 *)params, 1,
                &prevEvent, EDT_PROP_FINISH, NULL, &finishEvent);
        prevEvent = finishEvent;
    }

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &prevEvent,
            EDT_PROP_NONE, NULL, NULL);

    ocrEventSatisfy(initEvent, NULL_GUID);

    return NULL_GUID;
}

}
