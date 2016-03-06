#ifndef _HCLIB_LOCALITY_GRAPH_H
#define _HCLIB_LOCALITY_GRAPH_H

typedef struct _hclib_locale {
    unsigned id;
    char *lbl;
} hclib_locale;

typedef struct _hclib_locality_graph {
    hclib_locale *locales;
    unsigned n_locales;
    unsigned *edges;
} hclib_locality_graph;

typedef struct _hclib_locality_path {
    hclib_locale **locales;
    unsigned path_length;
} hclib_locality_path;

typedef struct _hclib_worker_paths {
    hclib_locality_path *pop_path;
    hclib_locality_path *steal_path;
} hclib_worker_paths;

extern void load_locality_info(char *filename, int *nworkers_out,
        hclib_locality_graph **graph_out,
        hclib_worker_paths **worker_paths_out);
extern void print_locality_graph(hclib_locality_graph *graph);
extern void print_worker_paths(hclib_worker_paths *worker_paths, int nworkers);

#endif
