/*
 * hclib-place.h
 *  
 *      Author: Vivek Kumar (vivekk@rice.edu)
 */

#ifndef HCLIB_PLACE_H_
#define HCLIB_PLACE_H_

#ifdef HC_CUDA
#include <driver_types.h>
#endif

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
extern place_t **hclib_get_nvgpu_places(int *n_nvgpu_places);
extern char *hclib_get_place_name(place_t *pl);

/*
 * hclib_allocate_at allocates a chunk of memory at a given place.
 * hclib_allocate_at is a blocking call, it does not offload the actual
 * allocation to the place specified. This allocation is run on the current
 * worker thread.
 */

extern void *hclib_allocate_at(place_t *pl, size_t nbytes, int flags);
/*
 * hclib_free_at releases the memory in place pl designated by memory alocation
 * ptr. Like hclib_allocate_at, this is a blocking call that runs on the current
 * thread.
 */
extern void hclib_free_at(place_t *pl, void *ptr);

#ifdef HC_CUDA

extern hclib_promise_t *hclib_async_copy(place_t *dst_pl, void *dst,
        place_t *src_pl, void *src, size_t nbytes,
        hclib_promise_t **promise_list, void *user_arg);
extern void hclib_async_copy_helper(place_t *dst_pl, void *dst, place_t *src_pl,
        void *src, size_t nbytes, hclib_promise_t **promise_list,
        void *user_arg, hclib_promise_t *out_promise);
extern hclib_promise_t *hclib_async_memset(place_t *pl, void *ptr, int val,
        size_t nbytes, hclib_promise_t **promise_list, void *user_arg);
extern void hclib_async_memset_helper(place_t *pl, void *ptr, int val,
        size_t nbytes, hclib_promise_t **promise_list, void *user_arg,
        hclib_promise_t *out_promise);
#endif

inline short is_cpu_place(place_t * pl) {
    HASSERT(pl);
    return (pl->type == MEM_PLACE || pl->type == CACHE_PLACE);
}

#ifdef HC_CUDA
inline short is_nvgpu_place(place_t * pl) {
    HASSERT(pl);
    return (pl->type == NVGPU_PLACE);
}
#endif

#endif /* HCLIB_PLACE_H_ */
