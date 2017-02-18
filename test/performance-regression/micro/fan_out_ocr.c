#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "fan_out.h"

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
    ocrGuid_t templateGuid = *((ocrGuid_t *)paramv);

    start_time = hclib_current_time_ns();

    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

    int i;
    for (i = 0; i < FAN_OUT; i++) {
        ocrGuid_t task;
        ocrEdtCreate(&task, templateGuid, 0, NULL, 1, &initEvent, EDT_PROP_NONE,
                NULL, NULL);
    }

    ocrEventSatisfy(initEvent, NULL_GUID);

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC fan_out %d %.20f\n", FAN_OUT,
            (double)FAN_OUT / ((double)(end_time - start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t templateGuid;
    ocrEdtTemplateCreate(&templateGuid, taskEdt, 0, 1);

    ocrGuid_t finishTemplateGuid;
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 1, 1);

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

    ocrGuid_t finishTask, finishTaskEvent;
    ocrEdtCreate(&finishTask, finishTemplateGuid, 1, (u64 *)&templateGuid,
            1, &initEvent, EDT_PROP_FINISH, NULL, &finishTaskEvent);

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &finishTaskEvent,
            EDT_PROP_NONE, NULL, NULL);

    ocrEventSatisfy(initEvent, NULL_GUID);

    return NULL_GUID;
}

}
