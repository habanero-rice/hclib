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

#ifndef HCLIB_COMMON_H_
#define HCLIB_COMMON_H_

#include "hclib_config.h"

/*
 * Default async arguments
 */

/** @brief No properties defined. */
#define NO_PROP 0
/** @brief No arguments provided. */
#define NO_ARG NULL
/** @brief To satisfy a promise with a 'NULL' value. */
#define NO_DATUM NULL
/** @brief No promse argument provided. */
#define NO_FUTURE NULL
/** @brief No phaser argument provided. */
#define NO_PHASER NULL
/** @brief No provided place, location is up to the runtime */
#define ANY_PLACE NULL
/** @brief To indicate an async must register with all phasers. */
#define PHASER_TRANSMIT_ALL ((int) 0x1) 
/** @brief No accumulator argument provided. */
#define NO_ACCUM NULL

#define HCLIB_LITECTX_STRATEGY 1
// #define VERBOSE 1

#endif
