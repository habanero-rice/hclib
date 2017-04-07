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
 * DESC: Top-level async spawn
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.h"

int ran = 0;

void async_fct(void * arg) {
    fprintf(stderr, "Running Async\n");
    ran = 1;
}

void entrypoint(void *arg) {
    fprintf(stderr, "Hello\n");

    hclib_start_finish();
    hclib_async(async_fct, NULL, NO_FUTURE, NO_PHASER, ANY_PLACE, NO_PROP);
    hclib_end_finish();

    assert(ran == 1);
    printf("Passed\n");
}

int main (int argc, char ** argv) {
    hclib_launch(entrypoint, NULL);
    fprintf(stderr, "Finished launch\n");
    return 0;
}
