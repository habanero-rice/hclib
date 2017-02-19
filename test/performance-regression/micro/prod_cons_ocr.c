#include "hclib.h"

#define ENABLE_EXTENSION_RTITF

#include <ocr.h>
#include <extensions/ocr-runtime-itf.h>

#include <stdio.h>
#include "prod_cons.h"

/*
 * Calculate micro-statistics:
 *
 *   1) Rate at which we can spawn empty tasks.
 *   2) Rate at which we can schedule and execute empty tasks.
 */

extern "C" {

static unsigned long long start_time = 0;

ocrGuid_t producerEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(paramc == 0);
    assert(depc == 1);

    ocrGuid_t dbGuid;
    int *dbPtr;
    ocrDbCreate(&dbGuid, (void **)&dbPtr, sizeof(*dbPtr), DB_PROP_NONE, NULL,
            NO_ALLOC);

    *dbPtr = 0;

    return dbGuid;
}

ocrGuid_t consumerEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(paramc == 0);
    assert(depc == 1);

    return NULL_GUID;
}

// Finish EDT
ocrGuid_t finishEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 0);
    assert(paramc == 2);
    ocrGuid_t *params = (ocrGuid_t *)paramv;
    ocrGuid_t producerTemplateGuid = params[0];
    ocrGuid_t consumerTemplateGuid = params[1];

    start_time = hclib_current_time_ns();

    ocrGuid_t initEvent;
    ocrEventCreate(&initEvent, OCR_EVENT_ONCE_T, EVT_PROP_NONE);

    int i;
    for (i = 0; i < PROD_CONS_MSGS; i++) {
        ocrGuid_t producer, producerEvent;
        ocrEdtCreate(&producer, producerTemplateGuid, 0, NULL, 1, &initEvent,
                EDT_PROP_NONE, NULL, &producerEvent);

        ocrGuid_t consumer;
        ocrEdtCreate(&consumer, consumerTemplateGuid, 0, NULL, 1,
                &producerEvent, EDT_PROP_NONE, NULL, NULL);
    }

    ocrEventSatisfy(initEvent, NULL_GUID);

    return NULL_GUID;
}

ocrGuid_t finalEdt(u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(depc == 1); // Triggered after runEdt finish EDT completes
    assert(paramc == 0);
    const unsigned long long end_time = hclib_current_time_ns();

    PRINTF("METRIC producer_consumer %d %.20f\n", PROD_CONS_MSGS,
            (double)PROD_CONS_MSGS / ((double)(end_time - start_time) /
                1000.0));

    ocrShutdown();

    return NULL_GUID;
}

ocrGuid_t mainEdt ( u32 paramc, u64* paramv, u32 depc, ocrEdtDep_t depv[]) {
    assert(sizeof(ocrGuid_t) == 8); // 64-bit GUIDs

    int nthreads = ocrNbWorkers();
    PRINTF("Using %d OCR threads\n", nthreads);

    ocrGuid_t producerTemplateGuid, consumerTemplateGuid, finishTemplateGuid,
              finalTemplateGuid;
    ocrEdtTemplateCreate(&producerTemplateGuid, producerEdt, 0, 1);
    ocrEdtTemplateCreate(&consumerTemplateGuid, consumerEdt, 0, 1);
    ocrEdtTemplateCreate(&finishTemplateGuid, finishEdt, 2, 0);
    ocrEdtTemplateCreate(&finalTemplateGuid, finalEdt, 0, 1);

    ocrGuid_t params[] = { producerTemplateGuid, consumerTemplateGuid };

    ocrGuid_t finishTask, finishTaskEvent;
    ocrEdtCreate(&finishTask, finishTemplateGuid, 2, (u64 *)params,
            0, NULL, EDT_PROP_FINISH, NULL, &finishTaskEvent);

    ocrGuid_t finalTask;
    ocrEdtCreate(&finalTask, finalTemplateGuid, 0, NULL, 1, &finishTaskEvent,
            EDT_PROP_NONE, NULL, NULL);

    return NULL_GUID;
}

}
