#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "hcpp.h"

int main(int argc, char ** argv) {	
	hcpp::init(&argc, argv);
	int numPlaces = hcpp::hc_get_num_places(hcpp::CACHE_PLACE);
	assert(numPlaces == 2);
	hcpp::place_t ** cachePlaces = (hcpp::place_t**) malloc(sizeof(hcpp::place_t*) * numPlaces);
	hcpp::hc_get_places(cachePlaces, hcpp::CACHE_PLACE);

	hcpp::place_t * p1 = cachePlaces[0];
	hcpp::place_t * p2 = cachePlaces[1];

	hcpp::finish([=]() {
		hcpp::asyncAt(p1, [=]() {
			assert(hcpp::current_ws()->pl == p1);			
		});

		hcpp::asyncAt(p2, [=]() {
			assert(hcpp::current_ws()->pl == p2);
		});
	});
	hcpp::finalize();
	return 0;
}
