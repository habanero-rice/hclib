/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * The source code in this file is based on similar code
 * provided as part of the UTS (Unbalanced Tree Search) benchmark:
 *
 * https://sourceforge.net/p/uts-benchmark/code/ci/722e89/tree/stats.c
 *
 * Please see the accompanying NOTICE file for license details.
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "hclib-timer.h"

#ifdef _TIMER_ON_
typedef struct stats_t {
    double time[HCLIB_NSTATES];	/* Time spent in each state */
    double timeLast;
    int    entries[HCLIB_NSTATES]; /* Num sessions of each state */
    int    curState;
} stats_t;

static stats_t *status;
static int numWorkers = -1;
static int have_comm_worker = 0;
#endif
double avgtime_nstates[HCLIB_NSTATES];

inline double wctime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void hclib_initStats  (int nw, int comm_w) {
#ifdef _TIMER_ON_
    numWorkers = nw;
    have_comm_worker = comm_w;
    status = new stats_t[numWorkers];
    for(int i=0; i<numWorkers; i++) {
        status[i].timeLast = wctime();
        status[i].curState = HCLIB_IDLE;
        for (int j = 0; j < HCLIB_NSTATES; j++) {
            status[i].time[j] = 0.0;
            status[i].entries[j] = 0;
        }
    }
#endif
}

/* Change states */
void hclib_setState(int wid, int state) {
#ifdef _TIMER_ON_
    double time;
    if (state < 0 || state >= HCLIB_NSTATES) {
        printf("ERROR: hclib_setState: thread state out of range");
        exit(-1);
    }
    if (state == status[wid].curState)
        return;

    time = wctime();
    status[wid].time[status[wid].curState] +=  time - status[wid].timeLast;
    status[wid].entries[state]++;
    status[wid].timeLast = time;
    status[wid].curState = state;
#endif
}

void find_avgtime_nstates() {
#ifdef _TIMER_ON_
    int start = have_comm_worker ? 1 : 0;
    int total = have_comm_worker ? (numWorkers - 1) : numWorkers;
    for(int j=0; j<HCLIB_NSTATES; j++) {
        avgtime_nstates[j] = 0;
        for(int i = start; i<numWorkers; i++) {
            avgtime_nstates[j] += status[i].time[j];
        }
        avgtime_nstates[j] = avgtime_nstates[j] / total;
    }
#endif
}

void hclib_get_avg_time(double *tWork, double *tOvh, double *tSearch) {
#ifdef _TIMER_ON_
    find_avgtime_nstates();
    *tWork = avgtime_nstates[HCLIB_WORK];
    *tSearch = avgtime_nstates[HCLIB_SEARCH];
    *tOvh = avgtime_nstates[HCLIB_OVH];
#else
    *tWork = 0;
    *tOvh = 0;
    *tSearch = 0;
#endif
}


