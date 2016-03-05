/*
 * hclib-timer.cpp
 *
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Variant of the implementation of hclib_setState() from the open source
 * implementation of UTS benchmark: http://sourceforge.net/projects/uts-benchmark/files/
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
static int start_worker = -1; // inclusive
static int end_worker = -1; // exclusive

#endif
double avgtime_nstates[HCLIB_NSTATES];

inline double wctime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void hclib_init_stats(int set_start_worker, int set_end_worker) {
#ifdef _TIMER_ON_
    start_worker = set_start_worker;
    end_worker = set_end_worker;

    status = new stats_t[end_worker - start_worker];
    for(int i = 0; i < end_worker - start_worker; i++) {
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
void hclib_set_state(int wid, int state) {
#ifdef _TIMER_ON_
    HASSERT(wid >= start_worker && wid < end_worker);
    HASSERT(state >= 0 && state < HCLIB_NSTATES);
    const int worker_offset = wid - start_worker;

    double time;
    if (state == status[worker_offset].curState) {
        return;
    }

    time = wctime();
    status[worker_offset].time[status[worker_offset].curState] +=
        time - status[worker_offset].timeLast;
    status[worker_offset].entries[state]++;
    status[worker_offset].timeLast = time;
    status[worker_offset].curState = state;
#endif
}

void find_avgtime_nstates() {
#ifdef _TIMER_ON_
    int start = have_comm_worker ? 1 : 0;
    int total = have_comm_worker ? (numWorkers - 1) : numWorkers;
    for(int j = 0; j < HCLIB_NSTATES; j++) {
        avgtime_nstates[j] = 0;
        for(int i = start; i<numWorkers; i++) {
            avgtime_nstates[j] += status[i].time[j];
        }
        avgtime_nstates[j] = avgtime_nstates[j] / total;
    }
#endif
}

void hclib_get_avg_time(double *t_work, double *t_ovh, double *t_search) {
#ifdef _TIMER_ON_
    find_avgtime_nstates();
    *t_work = avgtime_nstates[HCLIB_WORK];
    *t_search = avgtime_nstates[HCLIB_SEARCH];
    *t_ovh = avgtime_nstates[HCLIB_OVH];
#else
    *t_work = 0;
    *t_ovh = 0;
    *t_search = 0;
#endif
}


