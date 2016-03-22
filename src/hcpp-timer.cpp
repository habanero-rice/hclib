/*
 * hcpp-timer.cpp
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Variant of the implementation of hcpp_setState() from the open source
 * implementation of UTS benchmark: http://sourceforge.net/projects/uts-benchmark/files/
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "hcpp-timer.h"

namespace hcpp {

typedef struct stats_t {
	double time[HCPP_NSTATES];	/* Time spent in each state */
	double timeLast;
	int    entries[HCPP_NSTATES]; /* Num sessions of each state */
	int    curState;
} stats_t;

static stats_t* status;
double avgtime_nstates[HCPP_NSTATES];
static int numWorkers = -1;
static bool comm_worker = false;

inline double wctime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void hcpp_initStats  (int nw, bool comm_w) {
#ifdef _TIMER_ON_
	numWorkers = nw;
	comm_worker = comm_w;
	status = new stats_t[numWorkers];
	for(int i=0; i<numWorkers; i++) {
		status[i].timeLast = wctime();
		status[i].curState = HCPP_IDLE;
		for (int j = 0; j < HCPP_NSTATES; j++) {
			status[i].time[j] = 0.0;
			status[i].entries[j] = 0;
		}
	}
#endif
}

/* Change states */
void hcpp_setState(int wid, int state) {
#ifdef _TIMER_ON_
	double time;
	if (state < 0 || state >= HCPP_NSTATES) {
		printf("ERROR: hcpp_setState: thread state out of range");
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
	int start = comm_worker ? 1 : 0;
	int total = comm_worker ? (numWorkers - 1) : numWorkers;
	for(int j=0; j<HCPP_NSTATES; j++) {
		avgtime_nstates[j] = 0;
		for(int i = start; i<numWorkers; i++) {
			avgtime_nstates[j] += status[i].time[j];
		}
		avgtime_nstates[j] = avgtime_nstates[j] / total;
	}
#endif
}

void hcpp_getAvgTime (double* tWork, double *tOvh, double* tSearch) {
#ifdef _TIMER_ON_
	find_avgtime_nstates();
	*tWork = avgtime_nstates[HCPP_WORK];
	*tSearch = avgtime_nstates[HCPP_SEARCH];
	*tOvh = avgtime_nstates[HCPP_OVH];
#else
	*tWork = 0;
	*tOvh = 0;
	*tSearch = 0;
#endif
}

}
