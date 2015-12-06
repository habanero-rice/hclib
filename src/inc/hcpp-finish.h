#ifndef HCPP_FINISH_H
#define HCPP_FINISH_H

typedef struct finish_t {
    struct finish_t* parent;
    volatile int counter;
} finish_t;

#endif
