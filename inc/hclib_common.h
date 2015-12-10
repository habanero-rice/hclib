#ifndef HCLIB_COMMON_H_
#define HCLIB_COMMON_H_

/*
 * Default async arguments
 */

/** @brief No properties defined. */
#define NO_PROP 0
/** @brief No arguments provided. */
#define NO_ARG NULL
/** @brief To satisfy a DDF with a 'NULL' value. */
#define NO_DATUM NULL
/** @brief No DDF argument provided. */
#define NO_DDF NULL
/** @brief No phaser argument provided. */
#define NO_PHASER NULL
/** @brief To indicate an async must register with all phasers. */
#define PHASER_TRANSMIT_ALL ((int) 0x1) 
/** @brief No accumulator argument provided. */
#define NO_ACCUM NULL

#define HCLIB_LITECTX_STRATEGY 1
#define VERBOSE 1

#endif
