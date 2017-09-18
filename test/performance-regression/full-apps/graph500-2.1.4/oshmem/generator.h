#ifndef _GENERATOR_H
#define _GENERATOR_H

#include <stdint.h>

#include "packed_edge.h"
#include "mrg.h"

#ifdef __cplusplus
extern "C" {
#endif

void generate_kronecker_range(
        const uint_fast32_t seed[5] /* All values in [0, 2^31 - 1), not all zero */,
        int logN /* In base 2 */,
        int64_t start_edge, int64_t end_edge,
        packed_edge* edges);

#ifdef __cplusplus
}
#endif

#endif
