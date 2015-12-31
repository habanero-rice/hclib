/*
 * hcpp-timer.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Variant of the implementation of ss_setState() from the open source
 * implementation of UTS benchmark: http://sourceforge.net/projects/uts-benchmark/files/
 */

#ifndef HCPP_TIMER_H_
#define HCPP_TIMER_H_

/*
 * Comment this if you don't want timing analysis
 */
//#define _TIMER_ON_

/* Search status */
#define STATUS_HAVEWORK 0
#define STATUS_TERM     1

/* Search states */
#define HCPP_WORK    0
#define HCPP_SEARCH  1
#define HCPP_OVH     2
#define HCPP_IDLE    3
#define HCPP_NSTATES 4

void hcpp_initStats  (int numWorkers, int is_comm_worker);
void hcpp_setState   (int wid, int state);
void hclib_get_avg_time (double* tWork, double *tOvh, double* tSearch);

#define MARK_BUSY(w)	hcpp_setState(w, HCPP_WORK);
#define MARK_OVH(w)		hcpp_setState(w, HCPP_OVH);
#define MARK_SEARCH(w)	hcpp_setState(w, HCPP_SEARCH);

#endif /* HCPP_TIMER_H_ */
