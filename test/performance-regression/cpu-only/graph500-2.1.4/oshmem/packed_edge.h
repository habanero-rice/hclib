#ifndef _PACKED_EDGE_H
#define _PACKED_EDGE_H

#include <stdint.h>

typedef struct packed_edge {
    int64_t v0;
    int64_t v1;
} packed_edge;

static inline int64_t get_v0_from_edge(const packed_edge* p) {
    return p->v0;
}

static inline int64_t get_v1_from_edge(const packed_edge* p) {
    return p->v1;
}

static inline void write_edge(packed_edge* p, int64_t v0, int64_t v1) {
    p->v0 = v0;
    p->v1 = v1;
}

#endif
