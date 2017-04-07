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

#include <hclib_cpp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    hclib::launch([=] {
        printf("I see argc = %d, argv contains %s\n", argc, argv[0]);
        assert(argc == 1);
        assert(strcmp(argv[0], "./access_argc") == 0);
        printf("Check results: OK\n");
    });
    return 0;
}
