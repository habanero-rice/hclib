#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <shmem.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#include "mrg.h"
#include "packed_edge.h"
#include "utilities.h"
#include "generator.h"

#define INCOMING_MAILBOX_SIZE 33554432
#define OUTGOING_MAILBOX_SIZE 2097152

#define BITS_PER_INT 32

// #define VERBOSE
// #define PROFILE

static int pe = -1;
static int npes = -1;

uint64_t bfs_roots[] = {240425174, 115565041, 66063943, 180487911, 11178951,
    123935973, 231036167, 373595937, 363787030, 85801485, 108275987, 69071368,
    514373733, 251500048, 140103887, 506907254, 39995468, 195903646, 21863341,
    390997409, 470978452, 372755572, 449581394, 461086083, 357027875, 355651295,
    18628407, 427844427, 273604491, 372475785, 427329960, 465597328, 78313325,
    90706091, 457847627, 430362844, 178489195, 374418701, 7644678, 154891942,
    353689376, 56388509, 191747720, 264370699, 20638787, 421731131, 14127289,
    411537113, 397525451, 189929616, 140277533, 221845716, 135921328, 141538717,
    264336150, 267866811, 413698500, 263044574, 490922152, 81101617, 415841963,
    132009584, 67293842, 148419562};

volatile long long n_local_edges = 0;
volatile long long max_n_local_edges;
long long pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
short pWrk_short[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
int pWrk_int[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[SHMEM_REDUCE_SYNC_SIZE];

static uint64_t get_vertices_per_pe(uint64_t nvertices) {
    return (nvertices + npes - 1) / npes;
}

static uint64_t get_starting_vertex_for_pe(int pe, uint64_t nvertices) {
    uint64_t vertices_per_pe = get_vertices_per_pe(nvertices);
    return pe * vertices_per_pe;
}

static uint64_t get_ending_vertex_for_pe(int pe, uint64_t nvertices) {
    uint64_t vertices_per_pe = get_vertices_per_pe(nvertices);
    uint64_t limit = (pe + 1) * vertices_per_pe;
    if (limit > nvertices) limit = nvertices;
    return limit;
}

static inline int get_owner_pe(uint64_t vertex, uint64_t nvertices) {
    uint64_t vertices_per_pe = get_vertices_per_pe(nvertices);
    return vertex / vertices_per_pe;
}

static inline void set_visited(const uint64_t global_vertex_id,
        unsigned *visited, const unsigned visited_length,
        const uint64_t local_min_vertex) {
    // const uint64_t local_vertex_id = global_vertex_id - local_min_vertex;

    const int word_index = global_vertex_id / BITS_PER_INT;
    assert(word_index < visited_length);
    const int bit_index = global_vertex_id % BITS_PER_INT;
    const int mask = (1 << bit_index);

    // __sync_fetch_and_or(visited + word_index, mask);
    visited[word_index] |= mask;
}

static inline int is_visited(const uint64_t global_vertex_id,
        const unsigned *visited, const size_t visited_length,
        const uint64_t local_min_vertex) {
    // const uint64_t local_vertex_id = global_vertex_id - local_min_vertex;

    const unsigned word_index = global_vertex_id / BITS_PER_INT;
    assert(word_index < visited_length);
    const int bit_index = global_vertex_id % BITS_PER_INT;
    const int mask = (1 << bit_index);

    if ((visited[word_index] & mask) > 0) {
        return 1;
    } else {
        return 0;
    }
}

/* Spread the two 64-bit numbers into five nonzero values in the correct
 * range. */
static void make_mrg_seed(uint64_t userseed1, uint64_t userseed2,
        uint_fast32_t* seed) {
  seed[0] = (userseed1 & 0x3FFFFFFF) + 1;
  seed[1] = ((userseed1 >> 30) & 0x3FFFFFFF) + 1;
  seed[2] = (userseed2 & 0x3FFFFFFF) + 1;
  seed[3] = ((userseed2 >> 30) & 0x3FFFFFFF) + 1;
  seed[4] = ((userseed2 >> 60) << 4) + (userseed1 >> 60) + 1;
}

static int compare_uint64_t(const void *a, const void *b) {
    const uint64_t *aa = (const uint64_t *)a;
    const uint64_t *bb = (const uint64_t *)b;

    if (*aa < *bb) {
        return -1;
    } else if (*aa == *bb) {
        return 0;
    } else {
        return 1;
    }
}

static inline void send_new_vertices(packed_edge *send_buf,
        const int size, const int target_pe, packed_edge *filling, int *filling_incr) {
    const int remote_offset = shmem_int_fadd(filling_incr, size, target_pe);
    assert(remote_offset + size <= INCOMING_MAILBOX_SIZE);
    shmem_char_put_nbi((char *)(filling + remote_offset),
            (const char *)send_buf, sizeof(packed_edge) * size,
            target_pe);
}

static inline void handle_new_vertex(const uint64_t vertex, uint64_t *preds,
        packed_edge *reading, const unsigned *local_vertex_offsets,
        const uint64_t *neighbors, const uint64_t vertices_per_pe,
        packed_edge **send_bufs, unsigned *send_bufs_size,
        short *nmessages_local, unsigned *visited,
        const uint64_t local_min_vertex, const uint64_t local_max_vertex,
        const size_t visited_ints) {
    int i;
    assert(vertex >= local_min_vertex && vertex < local_max_vertex);
    const int local_vertex_id = vertex - local_min_vertex;

    const int parent = get_v1_from_edge(&reading[i]);

    set_visited(vertex, visited, visited_ints, local_min_vertex);
    set_visited(parent, visited, visited_ints, local_min_vertex);

    if (preds[local_vertex_id] == 0) {
        preds[local_vertex_id] = parent + 1;

        const int neighbor_start = local_vertex_offsets[local_vertex_id];
        const int neighbor_end = local_vertex_offsets[local_vertex_id + 1];

        int k;
        for (k = neighbor_start; k < neighbor_end; k++) {
            const uint64_t to_explore = neighbors[k];

            // Don't go backwards to the parent or follow self-loops
            if (to_explore != parent && to_explore != vertex &&
                    !is_visited(to_explore, visited, visited_ints, local_min_vertex)) {
                const int target_pe = to_explore / vertices_per_pe;
                packed_edge *send_buf = send_bufs[target_pe];
                int curr_size = send_bufs_size[target_pe];

                assert(curr_size < OUTGOING_MAILBOX_SIZE);

                set_visited(to_explore, visited, visited_ints, local_min_vertex);

                write_edge(&send_buf[curr_size], to_explore, vertex);

                curr_size++;
                send_bufs_size[target_pe] = curr_size;
                *nmessages_local = 1;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s scale edgefactor [num-bfs-roots]\n",
                argv[0]);
        fprintf(stderr, "    scale = log_2(# vertices)\n");
        fprintf(stderr, "    edgefactor = .5 * (average vertex degree)\n");
        fprintf(stderr, "    num-bfs-roots = # of roots to build a tree from "
                "[optional]\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "    For scale, the Graph500 benchmark defines the "
                "following presets:\n");
        fprintf(stderr, "        toy    = 26\n");
        fprintf(stderr, "        mini   = 29\n");
        fprintf(stderr, "        small  = 32\n");
        fprintf(stderr, "        medium = 36\n");
        fprintf(stderr, "        large  = 39\n");
        fprintf(stderr, "        huge   = 42\n");
        fprintf(stderr, "    The standard choice for edgefactor is 16\n");
        return 1;
    }

    const uint64_t scale = atoi(argv[1]);
    const uint64_t edgefactor = atoi(argv[2]);
    const uint64_t nglobaledges = (uint64_t)(edgefactor << scale);
    const uint64_t nglobalverts = (uint64_t)(((uint64_t)1) << scale);

    shmem_init();

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    uint_fast32_t seed[5];
    uint64_t seed1 = 2, seed2 = 3;
    make_mrg_seed(seed1, seed2, seed);

    const uint64_t edges_per_pe = (nglobaledges + npes - 1) / npes;
    const uint64_t start_edge_index = pe * edges_per_pe;
    uint64_t nedges_this_pe = edges_per_pe;
    if (start_edge_index + nedges_this_pe > nglobaledges) {
        nedges_this_pe = nglobaledges - start_edge_index;
        if (nedges_this_pe < 0) nedges_this_pe = 0;
    }

    if (pe == 0) {
        fprintf(stderr, "%llu: %lu total vertices, %lu total edges, %d PEs, ~%lu edges per "
                "PE, ~%lu vertices per PE\n", current_time_ns(), nglobalverts, nglobaledges, npes,
                edges_per_pe, get_vertices_per_pe(nglobalverts));
    }

    uint64_t i;

    /*
     * Use the Graph500 utilities to generate a set of edges distributed across
     * PEs.
     */
    packed_edge *actual_buf = (packed_edge *)malloc(
            nedges_this_pe * sizeof(packed_edge));
    assert(actual_buf || nedges_this_pe == 0);
    generate_kronecker_range(seed, scale, start_edge_index,
            start_edge_index + nedges_this_pe, actual_buf);

    /*
     * Count the number of edge endpoints in actual_buf that are resident on
     * each PE.
     */
    long long *count_edges_shared_with_pe = (long long *)calloc(npes,
            sizeof(long long));
    assert(count_edges_shared_with_pe);
    for (i = 0; i < nedges_this_pe; i++) {
        int64_t v0 = get_v0_from_edge(actual_buf + i);
        int64_t v1 = get_v1_from_edge(actual_buf + i);
        int v0_pe = get_owner_pe(v0, nglobalverts);
        int v1_pe = get_owner_pe(v1, nglobalverts);
        count_edges_shared_with_pe[v0_pe] += 1;
        count_edges_shared_with_pe[v1_pe] += 1;
    }

    /*
     * Tell each PE how many edges you have to send it based on vertex
     * ownership.
     */
    long long *remote_offsets = (long long *)malloc(npes * sizeof(long long));
    assert(remote_offsets);
    for (i = 0; i < npes; i++) {
        remote_offsets[i] = shmem_longlong_fadd((long long int *)&n_local_edges,
                count_edges_shared_with_pe[i], i);
    }
    free(count_edges_shared_with_pe);

    shmem_barrier_all();

    // Just for fun, find maximum # edges each PE will have
    for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_longlong_max_to_all((long long int *)&max_n_local_edges,
            (long long int *)&n_local_edges, 1, 0, 0, npes, pWrk, pSync);

    if (pe == 0) {
        fprintf(stderr, "%llu: Max. # local edges = %lld\n", current_time_ns(),
                max_n_local_edges);
    }

    uint64_t local_min_vertex = get_starting_vertex_for_pe(pe, nglobalverts);
    uint64_t local_max_vertex = get_ending_vertex_for_pe(pe, nglobalverts);
    uint64_t n_local_vertices;
    if (local_min_vertex >= local_max_vertex) {
        n_local_vertices = 0;
    } else {
        n_local_vertices = local_max_vertex - local_min_vertex;
    }

    // Contains parent-id + 1 for each local vertex
    uint64_t *preds = (uint64_t *)calloc(n_local_vertices, sizeof(uint64_t));
    assert(preds);

    /*
     * Allocate buffers on each PE for storing all edges for which at least one
     * of the vertices of the edge is handled by this PE. This information will
     * be provided by other PEs.
     */
    packed_edge *local_edges = (packed_edge *)shmem_malloc(
            max_n_local_edges * sizeof(packed_edge));
    assert(local_edges);

    /*
     * Send out to each PE based on the vertices each owns, all edges that have
     * a vertix on that node. This means that vertices which have one vertix on
     * one node and one vertix on another will be sent to two different nodes.
     */
    for (i = 0; i < nedges_this_pe; i++) {
        int64_t v0 = get_v0_from_edge(actual_buf + i);
        int64_t v1 = get_v1_from_edge(actual_buf + i);
        int v0_pe = get_owner_pe(v0, nglobalverts);
        int v1_pe = get_owner_pe(v1, nglobalverts);
        shmem_putmem(local_edges + remote_offsets[v0_pe], actual_buf + i,
                sizeof(packed_edge), v0_pe);
        remote_offsets[v0_pe]++;
        shmem_putmem(local_edges + remote_offsets[v1_pe], actual_buf + i,
                sizeof(packed_edge), v1_pe);
        remote_offsets[v1_pe]++;
    }

    free(remote_offsets);

    shmem_barrier_all();

    free(actual_buf);

    unsigned *local_vertex_offsets = (unsigned *)calloc(
            (n_local_vertices + 1), sizeof(unsigned));
    assert(local_vertex_offsets);

    /*
     * Location i in local_vertex_offsets stores the number of endpoints in
     * local_edges that have locale vertix i as one of the endpoints. Hence, it
     * is the total number of edge endpoints that are vertix i.
     */
    for (i = 0; i < n_local_edges; i++) {
        packed_edge *edge = local_edges + i;
        int64_t v0 = get_v0_from_edge(edge);
        int64_t v1 = get_v1_from_edge(edge);
        assert(get_owner_pe(v0, nglobalverts) == pe ||
                get_owner_pe(v1, nglobalverts) == pe);

        if (get_owner_pe(v0, nglobalverts) == pe) {
            local_vertex_offsets[v0 - local_min_vertex]++;
        }
        if (get_owner_pe(v1, nglobalverts) == pe) {
            local_vertex_offsets[v1 - local_min_vertex]++;
        }
    }

    /*
     * After this loop, location i in local_vertex_offsets stores a global
     * offset for vertix i in a local list of all endpoints stored on this PE.
     * The total number of local endpoints is the number of endpoints on the
     * locally stored edges that are for a vertix assigned to this PE (where the
     * locally stored edges are defined to be all edges that have at least one
     * vertix on this node). The sum of all local endpoints (the value in acc
     * after this loop) must be >= n_local_edges because each local edge must
     * have at least one endpoint that is a vertix on this node, but
     * <= n_local_edges * 2 because each edge can have at most 2 endpoints that
     * are vertices on this node.
     */
    uint64_t acc = 0;
    for (i = 0; i < n_local_vertices; i++) {
        uint64_t new_acc = acc + local_vertex_offsets[i];
        local_vertex_offsets[i] = new_acc; // point to the last element
        acc = new_acc;
    }
    local_vertex_offsets[n_local_vertices] = acc;
    assert(acc >= n_local_edges && acc <= n_local_edges * 2);

    /*
     * In neighbors, for each local endpoint discovered above we store the
     * destination vertex for that endpoint. So, after this loop, given local
     * vertex i:
     * 
     *     - its global vertex ID would be local_min_vertex + i
     *     - the list of global vertix IDs it is attached to by edges starts at
     *       local_vertex_offsets[i] and ends at local_vertex_offsets[i + 1]
     */
    uint64_t *neighbors = (uint64_t *)malloc(acc * 2 * sizeof(uint64_t));
    assert(neighbors);
    for (i = 0; i < n_local_edges; i++) {
        packed_edge *edge = local_edges + i;
        int64_t v0 = get_v0_from_edge(edge);
        int64_t v1 = get_v1_from_edge(edge);

        if (get_owner_pe(v0, nglobalverts) == pe) {
            neighbors[local_vertex_offsets[v0 - local_min_vertex] - 1] = v1;
            local_vertex_offsets[v0 - local_min_vertex]--;
        }
        if (get_owner_pe(v1, nglobalverts) == pe) {
            neighbors[local_vertex_offsets[v1 - local_min_vertex] - 1] = v0;
            local_vertex_offsets[v1 - local_min_vertex]--;
        }
    }

    // Remove duplicate edges
    uint64_t writing_index = 0;
    for (i = 0; i < n_local_vertices; i++) {
        const unsigned start = local_vertex_offsets[i];
        const unsigned end = local_vertex_offsets[i + 1];
        assert(start <= end);

        local_vertex_offsets[i] = writing_index;

        qsort(neighbors + start, end - start, sizeof(*neighbors),
                compare_uint64_t);

        uint64_t reading_index = start;
        while (reading_index < end) {
            unsigned j = reading_index + 1;
            while (j < end && neighbors[j] == neighbors[reading_index]) {
                j++;
            }
            neighbors[writing_index++] = neighbors[reading_index];
            reading_index = j;
        }
    }
    local_vertex_offsets[n_local_vertices] = writing_index;
    neighbors = (uint64_t *)realloc(neighbors, writing_index *
            sizeof(uint64_t));
    assert(neighbors);

    // Just some double checking
    for (i = 0; i < n_local_vertices; i++) {
        const unsigned neighbors_start = local_vertex_offsets[i];
        const unsigned neighbors_end = local_vertex_offsets[i + 1];

        int j;
        for (j = neighbors_start; j < neighbors_end; j++) {
            if (neighbors[j] >= nglobalverts) {
                fprintf(stderr, "Invalid neighbor at i = %llu / %llu, j = %u "
                        "(%u -> %u)\n", i,
                        n_local_vertices, j, neighbors_start, neighbors_end);
                assert(0);
            }
        }
    }

#ifdef VERBOSE
    fprintf(stderr, "PE %d found %llu duplicate edges, total endpoints = %llu\n", pe,
            duplicates, total_endpoints);
#endif

    shmem_free(local_edges);

    // Inbox and outbox
    packed_edge *reading = (packed_edge *)shmem_malloc(
            INCOMING_MAILBOX_SIZE * sizeof(packed_edge));
    packed_edge *filling = (packed_edge *)shmem_malloc(
            INCOMING_MAILBOX_SIZE * sizeof(packed_edge));
    assert(reading && filling);

    int *reading_incr = (int *)shmem_malloc(sizeof(int));
    int *filling_incr = (int *)shmem_malloc(sizeof(int));
    assert(reading_incr && filling_incr);
    *reading_incr = 0;
    *filling_incr = 0;

    // Buffers to use to transmit next wave to each other PE, output only
    packed_edge **send_bufs = (packed_edge **)malloc(npes *
            sizeof(packed_edge *));
    assert(send_bufs);
    unsigned *send_bufs_size = (unsigned *)calloc(npes, sizeof(unsigned));
    assert(send_bufs_size);
    for (i = 0; i < npes; i++) {
        send_bufs[i] = (packed_edge *)malloc(
                OUTGOING_MAILBOX_SIZE * sizeof(packed_edge));
        assert(send_bufs[i]);
    }

    short *nmessages_local = (short *)shmem_malloc(sizeof(short));
    short *nmessages_global = (short *)shmem_malloc(sizeof(short));
    assert(nmessages_local && nmessages_global);

    const size_t visited_ints = ((nglobalverts + BITS_PER_INT - 1) /
            BITS_PER_INT);
    const size_t visited_bytes = visited_ints * sizeof(unsigned);
    unsigned *visited = (unsigned *)malloc(visited_bytes);
    assert(visited);

    const unsigned num_bfs_roots = 3;
    assert(num_bfs_roots <= sizeof(bfs_roots) / sizeof(bfs_roots[0]));

    unsigned run;
    for (run = 0; run < num_bfs_roots; run++) {
        memset(visited, 0x00, visited_bytes);
        memset(preds, 0x00, n_local_vertices * sizeof(uint64_t));
        memset(send_bufs_size, 0x00, npes * sizeof(unsigned));

        uint64_t root = bfs_roots[run];

        if (get_owner_pe(root, nglobalverts) == pe) {
            /*
             * Signal that this PE has received 1 item in its inbox, setting the
             * parent for vertex 0 to 0.
             */
            set_visited(root, visited, visited_ints, local_min_vertex);
            write_edge(&(reading[0]), root, root);
            *reading_incr = 1;
        }

        shmem_barrier_all();
        const unsigned long long start_bfs = current_time_ns();
        int iter = 0;
        const uint64_t vertices_per_pe = get_vertices_per_pe(nglobalverts);

#ifdef PROFILE
        unsigned long long accum_time = 0;
        unsigned long long send_time = 0;
        unsigned long long reduce_time = 0;
#endif

        while (1) {
#ifdef PROFILE
            const unsigned long long start_accum = current_time_ns();
#endif
            *nmessages_local = 0;

            const int N = *reading_incr;
            *reading_incr = 0;

            for (i = 0; i < N; i++) {
                const int vertex = get_v0_from_edge(&reading[i]);

                handle_new_vertex(vertex, preds, reading, local_vertex_offsets,
                        neighbors, vertices_per_pe, send_bufs, send_bufs_size,
                        nmessages_local, visited, local_min_vertex,
                        local_max_vertex, visited_ints);
            }

#ifdef PROFILE
            const unsigned long long start_sends = current_time_ns();
            accum_time += (start_sends - start_accum);
#endif

            for (i = 0; i < npes; i++) {
                const int target = (pe + i) % npes;
                packed_edge *send_buf = send_bufs[target];
                const unsigned send_size = send_bufs_size[target];

                if (send_size > 0) {
#ifdef VERBOSE
                    fprintf(stderr, "On iter %d, PE %d sending %d entries to %d "
                            "(%d vertices per PE)\n", iter, pe, send_size, target,
                            get_vertices_per_pe(nglobalverts));
#endif
                    send_new_vertices(send_buf, send_size, target,
                            filling, filling_incr);
                    send_bufs_size[target] = 0;
                }
            }

#ifdef PROFILE
            const unsigned long long start_reduce = current_time_ns();
            send_time += (start_reduce - start_sends);
#endif

            shmem_short_max_to_all(nmessages_global, nmessages_local, 1, 0, 0, npes,
                    pWrk_short, pSync);

#ifdef PROFILE
            const unsigned long long end_all = current_time_ns();
            reduce_time += (end_all - start_reduce);
#endif

            if (*nmessages_global == 0) {
                break;
            }

            packed_edge *tmp = reading;
            reading = filling;
            filling = tmp;

            int *tmp_incr = reading_incr;
            reading_incr = filling_incr;
            filling_incr = tmp_incr;

            shmem_barrier_all();

            iter++;
        }

        shmem_barrier_all();
        const unsigned long long end_bfs = current_time_ns();
        if (pe == 0) {
            fprintf(stderr, "BFS %d with root=%llu took %f ms, %d iters\n",
                    run, root, (double)(end_bfs - start_bfs) / 1000000.0,
                    iter + 1);
        }

        int count_preds = 0;
        for (i = 0; i < n_local_vertices; i++) {
            if (preds[i] > 0) count_preds++;
        }
#ifdef VERBOSE
#ifdef PROFILE
        fprintf(stderr, "PE %d found preds for %d / %d local vertices, %llu ms "
                "accumulating, %llu ms sending, %llu ms reducing\n", pe,
                count_preds, n_local_vertices, accum_time / 1000000,
                send_time / 1000000, reduce_time / 1000000);
#else
        fprintf(stderr, "PE %d found preds for %d / %d local vertices, %llu "
                "endpoints\n", pe, count_preds, n_local_vertices, total_endpoints);
#endif
#endif
    }

    shmem_finalize();

    return 0;
}
