/*
 *  RICE University
 *  Habanero Team
 *  
 *  This file is part of HC Test.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

/*
 * Create async await and enable them (by a put) in the 
 * reverse order they've been created.
 */
int main(int argc, char ** argv) {
    constexpr float fval = 4.25f;
    hclib::promise_t<float> p{};
    p.put(fval);
    float res = p.future().get();
    printf("Put %f, got %f\n", fval, res);
    assert(res == fval);
    return 0;
}

