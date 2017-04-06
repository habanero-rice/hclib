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

#include "hclib.hpp"
#include "hclib-place.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    hclib::launch([] () {

            hclib::finish([] () {

                int numWorkers = hclib::num_workers();
                cout << "Total Workers: " << numWorkers << endl;

                int numPlaces = hclib::get_num_places(place_type_t::CACHE_PLACE);
                place_t **cachePlaces = (place_t**) malloc(sizeof(place_t*) * numPlaces);
                hclib::get_places(cachePlaces, place_type_t::CACHE_PLACE);

                for (int i = 0; i < numPlaces; i++) {

                place_t *currentPlace = cachePlaces[i];
                if (currentPlace->nChildren != 0) {
                cout << "CachePlace with children found, skipping" << endl;
                continue;
                }

                hclib::async([=] () {
                    cout << "Hello I'm Worker " << hclib::current_worker() << " of " << numWorkers << "Workers" << endl;
                    });
                }


                free(cachePlaces);

            });
    });

    return 0;
}
