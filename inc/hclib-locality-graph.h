#ifndef _HCLIB_LOCALITY_GRAPH_H
#define _HCLIB_LOCALITY_GRAPH_H

#include "hclib-rt.h"

struct _hclib_deque_t;
struct _hclib_task_t;

/*
 * A locality graph defines the reachable hardware components from each locale
 * in a platform. It consists of a set of locales linked together by
 * bi-directional edges. Each locale should generally be associated with a
 * physical hardware component, though not necessarily a piece of the memory
 * hierarchy (as is the case with HPTs). For example, there can be locales for a
 * piece of L1 cache but there can also be locales for a network card to allow
 * the creation of work at that network card. At the moment locality graphs are
 * purely within a node, but this will probably change in the future.
 *
 * Given a defined locality graph for a platform, a worker path defines the path
 * through that locality graph that a certain worker thread takes when trying to
 * either 'pop' work that it has created or 'steal' work that other worker
 * threads have created. A worker path does not necessarily have to follow edges
 * in the locality graph, though doing so may have obvious locality benefits.
 *
 * These concepts generalize the ideas of communication threads, GPU threads,
 * PHI threads, etc that have been thrown around in the research for several
 * years now. A communication thread is simply a worker thread that includes an
 * interconnect locale in its list of pop or steal places. A GPU thread is a
 * worker thread that includes a GPU locale in its list of pop or steal places.
 *
 * Hand in hand with the idea of locality graphs and worker paths is the concept
 * of modules in HClib. A module is a plug-in (structured as a C++ header file)
 * that hooks into HClib to add the ability to do something in addition to
 * HClib's intrinsic dynamic tasking abilities. For example, an MPI module would
 * wrap an MPI library and might add calls such as MPI_Send, MPI_Recv, etc to
 * the hclib namespace. Under the covers, these calls would place tasks at
 * interconnect locales to later be serviced by threads that include the
 * interconnect locale in their pop/steal path. The ability to create tasks at
 * different locales is not limited to any subset of workers, i.e. any worker
 * can create work at any locale.
 *
 * This design does introduce the ability of the user to create a locality graph
 * and locality paths such that no worker ever services certain locales, leading
 * to deadlock. That is a user error and is not currently addressed. It would be
 * possible in the future to verify that all locales in a system are covered by
 * at least one worker path, guaranteeing that any tasks being created would be
 * executed eventually. However, this might also cause inefficiencies if we know
 * work will never be created at a certain locale for a given program but still
 * must visit it along at least one locale path.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _hclib_locale_t {
    int id;
    unsigned type;
    const char *lbl;
    void *metadata;

    struct _hclib_deque_t *deques;
} hclib_locale_t;

typedef struct _hclib_locality_graph {
    hclib_locale_t *locales;
    unsigned n_locales;
    unsigned *edges;
} hclib_locality_graph;

typedef struct _hclib_locality_path {
    hclib_locale_t **locales;
    unsigned path_length;
} hclib_locality_path;

typedef struct _hclib_worker_paths {
    hclib_locality_path *pop_path;
    hclib_locality_path *steal_path;
} hclib_worker_paths;

extern void load_locality_info(const char *filename, int *nworkers_out,
        hclib_locality_graph **graph_out,
        hclib_worker_paths **worker_paths_out);
extern void generate_locality_info(int *nworkers_out,
        hclib_locality_graph **graph_out,
        hclib_worker_paths **worker_paths_out);
extern void check_locality_graph(hclib_locality_graph *graph,
        hclib_worker_paths *worker_paths, int nworkers);
extern void print_locality_graph(hclib_locality_graph *graph);
extern void print_worker_paths(hclib_worker_paths *worker_paths, int nworkers);
extern int deque_push_locale(hclib_worker_state *ws, hclib_locale_t *locale,
        void *ele);
extern struct _hclib_task_t *locale_pop_task(hclib_worker_state *ws);
extern struct _hclib_task_t *locale_steal_task(hclib_worker_state *ws);

extern int hclib_get_num_locales();
extern hclib_locale_t *hclib_get_closest_locale();
extern hclib_locale_t *hclib_get_master_place();
extern hclib_locale_t *hclib_get_all_locales();
extern hclib_locale_t *hclib_get_closest_locale_of_types(hclib_locale_t *locale,
        int *locale_types, int n_locale_types);
extern hclib_locale_t *hclib_get_closest_locale_of_type(hclib_locale_t *locale,
        int locale_type);
extern hclib_locale_t **hclib_get_all_locales_of_type(int type, int *out_count);
extern int hclib_get_num_locales_of_type(int locale_type);

extern unsigned hclib_add_known_locale_type(const char *lbl);

#ifdef __cplusplus
}
#endif

#endif
