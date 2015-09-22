#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hcpp.h"

int main(int argc, char ** argv) {	
	hcpp::init(&argc, argv);
	int numPlaces = hc_get_num_places(CACHE_PLACE);
	assert(numPlaces == 2);
	place_t ** cachePlaces = (place_t**) malloc(sizeof(place_t*) * numPlaces);
	hc_get_places(cachePlaces, CACHE_PLACE);

	place_t * p1 = cachePlaces[0];
	place_t * p2 = cachePlaces[1];

	hcpp::finish([=]() {
		hcpp::asyncAt(p1, [=]() {
			assert(current_ws()->pl == p1);			
		});

		hcpp::asyncAt(p2, [=]() {
			assert(current_ws()->pl == p2);
		});
	});
	hcpp::finalize();
	return 0;
}
