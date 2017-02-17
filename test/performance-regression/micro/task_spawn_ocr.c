#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "task_spawn.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

ocrGuid_t taskEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    int incr = 0;
    incr = incr + 1;
    ocrGuid_t ret = { 0 };
    return ret;
}

static void driver(const ocrGuid_t templateGuid) {
    int nlaunched = 0;
    do {
        ocrGuid_t task;
        ocrEdtCreate(&task, templateGuid, EDT_PARAM_DEF, NULL, EDT_PARAM_DEF,
                NULL, EDT_PROP_NONE, NULL, NULL);

        nlaunched++;
    } while (nlaunched < NTASKS);
}

// Finish EDT
ocrGuid_t createEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 1);
    ocrGuid_t templateGuid = *((ocrGuid_t *)paramv);

    const unsigned long long spawn_start_time = hclib_current_time_ns();

    driver(templateGuid);

    const unsigned long long spawn_end_time = hclib_current_time_ns();
    PRINTF("METRIC task_create %d %f\n", NTASKS,
            (double)NTASKS / ((double)(spawn_end_time -
                    spawn_start_time) / 1000.0));

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t runEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after createEdt finish EDT completes
    assert(paramc == 1);
    ocrGuid_t templateGuid = *((ocrGuid_t *)paramv);
    unsigned long long schedule_start_time;

    ocrGuid_t start_time_block;
    void *start_time_block_addr;
    ocrDbCreate(&start_time_block, &start_time_block_addr,
            sizeof(schedule_start_time), DB_PROP_NONE, NULL, NO_ALLOC);

    *((unsigned long long *)start_time_block_addr) = hclib_current_time_ns();

    driver(templateGuid);

    return start_time_block;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long schedule_end_time = hclib_current_time_ns();
    const unsigned long long schedule_start_time = *((unsigned long long *)depv[0].ptr);

    PRINTF("METRIC task_run %d %f\n", NTASKS,
            (double)NTASKS / ((double)(schedule_end_time -
                    schedule_start_time) / 1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t templateGuid;
    ocrEdtTemplateCreate(&templateGuid, taskEdt, 0, 0);

    ocrGuid_t createTemplateGuid;
    ocrEdtTemplateCreate(&createTemplateGuid, createEdt, 1, 0);

    ocrGuid_t runTemplateGuid;
    ocrEdtTemplateCreate(&runTemplateGuid, runEdt, 1, 1);

    ocrGuid_t finalTemplateGuid;
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t spawnTask, spawnTaskEvent;
    ocrEdtCreate(&spawnTask, createTemplateGuid, 1, (u64 *)&templateGuid,
            0, NULL, EDT_PROP_FINISH, NULL, &spawnTaskEvent);

    ocrGuid_t runTask, runTaskEvent;
    ocrEdtCreate(&runTask, runTemplateGuid, 1, (u64 *)&templateGuid,
            1, &spawnTaskEvent, EDT_PROP_FINISH, NULL, &runTaskEvent);

    ocrGuid_t finalTask, finalTaskEvent;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL,
            1, &runTaskEvent, EDT_PROP_NONE, NULL, &finalTaskEvent);

    return NULL_GUID;
}

}
