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

void hclib_initStats  (int numWorkers);
void hclib_setState   (int wid, int state);
void hclib_get_avg_time (double* tWork, double *tOvh, double* tSearch);

#define MARK_BUSY(w)	hclib_setState(w, HCLIB_WORK);
#define MARK_OVH(w)		hclib_setState(w, HCLIB_OVH);
#define MARK_SEARCH(w)	hclib_setState(w, HCLIB_SEARCH);

#endif /* HCLIB_TIMER_H_ */
