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

#ifndef HCLIB_WORKER_CONFIG_H_
#define HCLIB_WORKER_CONFIG_H_

#include <stdint.h>

/*
 * NOTE:
 *
 * This file is assumed to follow a strict format to allow easy parsing
 * by the hclib-options script. A list of the most important assumptions
 * is included below. See the script for exact details.
 *
 * - Lines starting with C++-style double-slash comments are assumed
 *   to be part of a description of a worker configuration option.
 *
 * - Definitions with the prefix HCLIB_WORKER_STRATEGY_ are assumed
 *   to be options for setting the worker context-management strategy.
 *
 * - Definitions with the prefix HCLIB_WORKER_OPTIONS_ are assumed
 *   to be additional options for the selected worker strategy.
 *
 * - Values for both types of definitions are assumed to be int constants.
 */


/***************************************/
/* runtime worker threading strategies */
/***************************************/

// Fixed pool of worker threads.
#define HCLIB_WORKER_STRATEGY_FIXED    0x01

// Spawn a new fiber for each blocked task,
// and join a fiber when a blocked task is resumed.
#define HCLIB_WORKER_STRATEGY_FIBERS   0x02

// Spawn a new thread for each blocked task,
// and join a thread when a blocked task is resumed.
#define HCLIB_WORKER_STRATEGY_THREADS  0x03


/************************************/
/* runtime worker threading options */
/************************************/

// Allow a worker context to run any ready task on top of the current
// execution stack when a task blocks to help with global progress.
// This option only affects the fixed worker pool strategy.
// MAY CAUSE DEADLOCKS!
#define HCLIB_WORKER_OPTIONS_HELP_GLOBAL  0x01

// Allow a worker context to run a ready task from the current
// finish scope on top of the current execution stack when blocking
// on an end-finish to help with that finish scope's progress.
#define HCLIB_WORKER_OPTIONS_HELP_FINISH  0x02

// Do not explicitly join worker threads;
// i.e., pthreads are created with the attribute PTHREAD_CREATE_DETACHED.
#define HCLIB_WORKER_OPTIONS_NO_JOIN      0x04


/*******************************/
/* set up strategy and options */
/*******************************/

#define HCLIB_DEFAULT_WORKER_STRATEGY HCLIB_WORKER_STRATEGY_FIBERS
#define HCLIB_DEFAULT_WORKER_OPTIONS  HCLIB_WORKER_OPTIONS_HELP_FINISH

#ifdef HCLIB_DEBUG
/* dynamic choice of worker configuration supported for debugging */
#define HCLIB_WORKER_STRATEGY hclib_worker_strategy()
#define HCLIB_WORKER_OPTIONS  hclib_worker_options()
#else
/* static choice of worker configuration required for production */
#ifdef HCLIB_SET_WORKER_STRATEGY
#define HCLIB_WORKER_STRATEGY HCLIB_SET_WORKER_STRATEGY
#define HCLIB_WORKER_OPTIONS  HCLIB_SET_WORKER_OPTIONS
#else
#define HCLIB_WORKER_STRATEGY HCLIB_DEFAULT_WORKER_STRATEGY
#define HCLIB_WORKER_OPTIONS  HCLIB_DEFAULT_WORKER_OPTIONS
#endif  /* HCLIB_SET_WORKER_STRATEGY */
#endif  /* HCLIB_DEBUG */


/**************************************/
/* functions to get the configuration */
/**************************************/

int64_t hclib_worker_strategy();
int64_t hclib_worker_options();

#endif  /* HCLIB_WORKER_CONFIG_H_ */
