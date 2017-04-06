#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hclib.hpp"

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
