#ifndef HCLIB_OPENSHMEM_INTERNAL_AM_H
#define HCLIB_OPENSHMEM_INTERNAL_AM_H

#include "hclib-module.h"
#include "hclib-locality-graph.h"
#include "hclib_cpp.h"

#include <shmem.h>
#include <shmemx.h>

#include <stdlib.h>
#include <stdio.h>

namespace hclib {

HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_pre_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_post_initialize);
HCLIB_MODULE_INITIALIZATION_FUNC(openshmem_am_finalize);

}

#endif
