/*
Copyright (c) 2015, Intel Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include "parse-args.h"

#include <shmem.h>
#include <cstdio>
#include <cinttypes>
#include <cassert>
#include <cmath>

#include "sort-common.hpp"


void parseArgsFlat(int &argc, char **&argv) {
    if (argc != 3) {
        if (shmem_my_pe() == 0) {
            printf("Usage:  \n");
            printf("  ./%s <total num keys(strong) | keys per pe(weak)> <log_file>\n", argv[0]);
        }

        shmem_finalize();
        exit(1);
    }

    NUM_PES = (uint64_t) shmem_n_pes();
    MAX_KEY_VAL = DEFAULT_MAX_KEY;
    NUM_BUCKETS = NUM_PES;
    BUCKET_WIDTH = (uint64_t) ceil((double) MAX_KEY_VAL / NUM_BUCKETS);
    char *log_file = argv[2];
    char scaling_msg[64];

    switch (SCALING_OPTION) {
        case STRONG: {
            TOTAL_KEYS = (uint64_t) atoi(argv[1]);
            NUM_KEYS_PER_PE = (uint64_t) ceil((double) TOTAL_KEYS / NUM_PES);
            sprintf(scaling_msg, "STRONG");
            break;
        }

        case WEAK: {
            NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
            sprintf(scaling_msg, "WEAK");
            break;
        }

        case WEAK_ISOBUCKET: {
            NUM_KEYS_PER_PE = (uint64_t) (atoi(argv[1]));
            BUCKET_WIDTH = ISO_BUCKET_WIDTH;
            MAX_KEY_VAL = (uint64_t) (NUM_PES * BUCKET_WIDTH);
            sprintf(scaling_msg, "WEAK_ISOBUCKET");
            break;
        }

        default: {
            if (shmem_my_pe() == 0) {
                printf("Invalid scaling option! See params.h to define the scaling option.\n");
            }

            shmem_finalize();
            exit(1);
            break;
        }
    }

    assert(MAX_KEY_VAL > 0);
    assert(NUM_KEYS_PER_PE > 0);
    assert(NUM_PES > 0);
    assert(MAX_KEY_VAL > NUM_PES);
    assert(NUM_BUCKETS > 0);
    assert(BUCKET_WIDTH > 0);

    if (shmem_my_pe() == 0) {
        printf("ISx v%1d.%1d\n", MAJOR_VERSION_NUMBER, MINOR_VERSION_NUMBER);
        printf("  Number of Keys per PE: %" PRIu64 "\n", NUM_KEYS_PER_PE);
        printf("  Max Key Value: %" PRIu64 "\n", MAX_KEY_VAL);
        printf("  Bucket Width: %" PRIu64 "\n", BUCKET_WIDTH);
        printf("  Number of Iterations: %u\n", NUM_ITERATIONS);
        printf("  Number of PEs: %" PRIu64 "\n", NUM_PES);
        printf("  %s Scaling!\n", scaling_msg);
    }

    //return log_file;
}
