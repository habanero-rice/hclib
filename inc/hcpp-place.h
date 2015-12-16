/*
 * hcpp-place.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#ifndef HCPP_PLACE_H_
#define HCPP_PLACE_H_

#include "hcpp-cuda.h"

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
#ifdef HC_CUDA
    int cuda_id;
    cudaStream_t cuda_stream;
#endif
} place_t ;

typedef enum place_alloc_flags {
    NONE = 0,
    PHYSICAL
} place_alloc_flags_t;

/** PLACES API **/
extern int hclib_get_num_places(short type);
extern void hclib_get_places(place_t ** pls, short type);
extern place_t * hc_get_place(short type);

extern place_t *hclib_get_root_place();
extern place_t *hclib_get_current_place();
extern place_t *hclib_get_child_place();
extern place_t *hclib_get_parent_place();
extern place_t **hclib_get_children_places(int * numChildren);
extern place_t **hclib_get_children_of_place(place_t * pl, int * numChildren);

extern void *hclib_allocate_at(place_t *pl, size_t nbytes, int flags);
extern void hclib_free_at(place_t *pl, void *ptr);
extern hclib_ddf_t *hclib_async_copy(place_t *dst_pl, void *dst,
        place_t *src_pl, void *src, size_t nbytes);

#endif /* HCPP_PLACE_H_ */
