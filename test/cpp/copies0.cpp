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

/**
 * DESC: Counting lambda copies for a simple async
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

struct CopyCounter {
    int copies;
    int moves;

    CopyCounter(): copies(0), moves(0) {
        printf("Created new counter object.\n");
    }

    CopyCounter(const CopyCounter &other):
        copies(other.copies+1), moves(other.moves) { }

    CopyCounter(const CopyCounter &&other):
        copies(other.copies), moves(other.moves+1) { }
};

int main (int argc, char ** argv) {
    hclib::launch([]() {
        hclib::finish([]() {
            CopyCounter k {};
            hclib::async([k] {
                printf("Counted %d copies, %d moves...\n", k.copies, k.moves);
                // Copy 1: capturing "k" by-value into the lambda
                // Copy 2: copying the lambda (and its closure) into the heap
                assert(k.copies + k.moves <= 2);
            });
        });
    });
    printf("Exiting...\n");
    return 0;
}
