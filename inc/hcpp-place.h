/*
 * hcpp-place.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#ifndef HCPP_PLACE_H_
#define HCPP_PLACE_H_

struct hc_deque_t;

typedef enum place_type {
	CACHE_PLACE,
	MEM_PLACE,
	NVGPU_PLACE, /* for Nvidia GPU */
	AMGPU_PLACE, /* for AMD GPU */
	FPGA_PLACE,  /* for FPGA */
	PGAS_PLACE,
	NUM_OF_TYPES,
} place_type_t;

typedef struct place_t {
	struct place_t * parent;
	struct place_t * child; /* the first child */
	struct place_t * nnext; /* the sibling link of the HPT */
	struct place_t ** children;
	struct hc_workerState * workers; /* directly attached cpu workers */
	struct hc_deque_t * deques;
	int ndeques; /* only for deques */
	int id;
	int did; /* the mapping device id */
	int unitSize;
	int psize;
	int level; /* Level in the HPT tree. Logical root is level 0. */
	int nChildren;
	short type;
} place_t ;

/** PLACES API **/
int hc_get_num_places(short type);
void hc_get_places(place_t ** pls, short type);
place_t * hc_get_place(short type);

place_t * hc_get_root_place();
place_t * hc_get_current_place();
place_t * hc_get_child_place();
place_t * hc_get_parent_place();
place_t ** hc_get_children_places(int * numChildren);
place_t ** hc_get_children_of_place(place_t * pl, int * numChildren);

#endif /* HCPP_PLACE_H_ */
