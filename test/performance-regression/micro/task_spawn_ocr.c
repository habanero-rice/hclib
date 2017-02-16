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

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    int nthreads = ocrNbWorkers();
    printf("Using %d OCR threads\n", nthreads);

    ocrGuid_t templateGuid;
    ocrEdtTemplateCreate(&templateGuid, taskEdt, 0, 1);

    ocrGuid_t trigger;
    ocrEventCreate(&trigger, OCR_EVENT_ONCE_T, EVT_PROP_TAKES_ARG);

    int nlaunched = 0;
    const unsigned long long spawn_start_time = hclib_current_time_ns();
    do {
        ocrGuid_t depv = trigger;

        ocrGuid_t task;
        ocrEdtCreate(&task, templateGuid, EDT_PARAM_DEF, NULL, EDT_PARAM_DEF,
                &depv, EDT_PROP_NONE, NULL, NULL);

        nlaunched++;
    } while (nlaunched < NTASKS);
    ocrEventSatisfy(trigger, NULL_GUID);

    const unsigned long long spawn_end_time = hclib_current_time_ns();
    printf("METRIC task_create %d %f\n", NTASKS,
            (double)NTASKS / ((double)(spawn_end_time -
                    spawn_start_time) / 1000.0));

    ocrShutdown();
    ocrGuid_t ret = { 0 };
    return ret;
}

}
