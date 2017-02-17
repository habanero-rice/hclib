#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "future_spawn.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long schedule_start_time = 0;

ocrGuid_t taskEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    int incr = 0;
    incr = incr + 1;
    ocrGuid_t ret = { 0 };
    return ret;
}

static void driver(const ocrGuid_t taskTemplateGuid,
        const ocrGuid_t futureTemplateGuid) {
    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);
    ocrGuid_t prevEvent = initEvent;

    int nlaunched = 0;
    do {
        ocrGuid_t task, taskEvent;
        ocrEdtCreate(&task, futureTemplateGuid, 0, NULL, 1,
                &prevEvent, EDT_PROP_NONE, NULL, &taskEvent);
        prevEvent = taskEvent;

        nlaunched++;
    } while (nlaunched < NFUTURES);

    ocrEventSatisfy(initEvent, NULL_GUID);
}

// Finish EDT
ocrGuid_t createEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 2);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t taskTemplateGuid = params[0];
    ocrGuid_t futureTemplateGuid = params[1];

    const unsigned long long spawn_start_time = hclib_current_time_ns();

    driver(taskTemplateGuid, futureTemplateGuid);

    const unsigned long long spawn_end_time = hclib_current_time_ns();
    PRINTF("METRIC future_create %d %.20f\n", NFUTURES,
            (double)NFUTURES / ((double)(spawn_end_time -
                    spawn_start_time) / 1000.0));

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t runEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after createEdt finish EDT completes
    assert(paramc == 2);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t taskTemplateGuid = params[0];
    ocrGuid_t futureTemplateGuid = params[1];

    schedule_start_time = hclib_current_time_ns();

    driver(taskTemplateGuid, futureTemplateGuid);

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long schedule_end_time = hclib_current_time_ns();

    PRINTF("METRIC future_run %d %.20f\n", NFUTURES,
            (double)NFUTURES / ((double)(schedule_end_time -
                    schedule_start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t taskTemplateGuid;
    ocrEdtTemplateCreate(&taskTemplateGuid, taskEdt, 0, 0);

    ocrGuid_t futureTemplateGuid;
    ocrEdtTemplateCreate(&futureTemplateGuid, taskEdt, 0, 1);

    ocrGuid_t params[] = { taskTemplateGuid, futureTemplateGuid };

    ocrGuid_t createTemplateGuid;
    ocrEdtTemplateCreate(&createTemplateGuid, createEdt, 2, 0);

    ocrGuid_t runTemplateGuid;
    ocrEdtTemplateCreate(&runTemplateGuid, runEdt, 2, 1);

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t spawnTask, spawnTaskEvent;
    ocrEdtCreate(&spawnTask, createTemplateGuid, 2, (u64 *)params,
            0, NULL, EDT_PROP_FINISH, NULL, &spawnTaskEvent);

    ocrGuid_t runTask, runTaskEvent;
    ocrEdtCreate(&runTask, runTemplateGuid, 2, (u64 *)params,
            1, &spawnTaskEvent, EDT_PROP_FINISH, NULL, &runTaskEvent);

    ocrGuid_t finalTask, finalTaskEvent;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL,
            1, &runTaskEvent, EDT_PROP_NONE, NULL, &finalTaskEvent);

    return NULL_GUID;
}

}
