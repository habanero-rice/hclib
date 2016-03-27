#ifndef HCLIB_SYSTEM_H
#define HCLIB_SYSTEM_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <stdlib.h>
#include <stdio.h>

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(system_initialize);

hclib::locale_t *get_closest_cpu_locale();

}

#endif
