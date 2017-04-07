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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib_cpp.h"

int main(int argc, char ** argv) {	
	hclib::launch(&argc, argv, [&]() {
        int numPlaces = hclib::get_num_places(hclib::place_type_t::CACHE_PLACE);
        assert(numPlaces == 3);
        hclib::place_t ** cachePlaces = (hclib::place_t**) malloc(
                sizeof(hclib::place_t*) * numPlaces);
        hclib::get_places(cachePlaces, hclib::place_type_t::CACHE_PLACE);

        hclib::place_t * p1 = cachePlaces[0];
        hclib::place_t * p2 = cachePlaces[1];

        hclib::finish([=]() {
            hclib::asyncAtHpt(p1, [=]() {
                assert(hclib::current_ws()->pl == p1);			
            });

            hclib::asyncAtHpt(p2, [=]() {
                assert(hclib::current_ws()->pl == p2);
            });
        });
    });
    printf("Passed");
	return 0;
}
