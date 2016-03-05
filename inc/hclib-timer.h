/*
 * hclib-timer.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

/*
 * Variant of the implementation of ss_setState() from the open source
 * implementation of UTS benchmark: http://sourceforge.net/projects/uts-benchmark/files/
 */

#ifndef HCLIB_TIMER_H_
#define HCLIB_TIMER_H_

/*
 * Comment this if you don't want timing analysis
 */
//#define _TIMER_ON_

/* Search status */
#define STATUS_HAVEWORK 0
#define STATUS_TERM     1

/* Search states */
#define HCLIB_WORK    0
#define HCLIB_SEARCH  1
#define HCLIB_OVH     2
#define HCLIB_IDLE    3
#define HCLIB_NSTATES 4

void hclib_init_stats(int start_worker, int end_worker);
void hclib_set_state(int wid, int state);
void hclib_get_avg_time (double* t_work, double *t_ovh, double* t_search);

#define MARK_BUSY(w)	hclib_set_state(w, HCLIB_WORK);
#define MARK_OVH(w)		hclib_set_state(w, HCLIB_OVH);
#define MARK_SEARCH(w)	hclib_set_state(w, HCLIB_SEARCH);

#endif /* HCLIB_TIMER_H_ */
