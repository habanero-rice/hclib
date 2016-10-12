#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <shmem.h>
#include <time.h>
#include <sys/time.h>

#include "mrg.h"
#include "packed_edge.h"
#include "utilities.h"
#include "generator.h"

static int pe = -1;
static int npes = -1;

volatile long long n_local_edges = 0;
volatile long long max_n_local_edges;
long long pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync[SHMEM_REDUCE_SYNC_SIZE];

int pWrk_int[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
long pSync_int[SHMEM_REDUCE_SYNC_SIZE];

#define UNSIGNALLED 0

volatile long long *nsignals = NULL;
int ndone = 0;
int consensus, vote;
int checking_if_done = 0;
int nstill_working = 0;

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

static int get_owner_pe(uint64_t vertex, uint64_t nvertices) {
    uint64_t vertices_per_pe = get_vertices_per_pe(nvertices);
    return vertex / vertices_per_pe;
}

/*
 * Returns the number of parent relationships successfully made with children of
 * any vertices in our current wavefront.
 */
static int scan_signals(uint64_t *min_unsignalled, uint64_t *max_unsignalled,
        uint64_t n_local_vertices,
        int *signals, unsigned *local_vertex_offsets, uint64_t *neighbors,
        uint64_t nglobalverts, const uint64_t local_min_vertex,
        const uint32_t iteration) {
    uint64_t i, j;

    // int *unique_verts = NULL;
    // uint64_t nunique_verts = 0;
    // int *unique_pes = NULL;
    // uint64_t nunique_pes = 0;
    uint64_t nfailed = 0;

    const uint64_t save_min_unsignalled = *min_unsignalled;
    const uint64_t save_max_unsignalled = *max_unsignalled;

    uint64_t new_min_unsignalled = save_min_unsignalled;
    uint64_t new_max_unsignalled = save_max_unsignalled;

    static long long int *pe_incrs = NULL;
    if (pe_incrs == NULL) {
        pe_incrs = (long long int *)malloc(npes * sizeof(long long int));
    }
    memset(pe_incrs, 0x00, npes * sizeof(long long int));

    int sent_any_signals = 0;
    for (i = 0; i < n_local_vertices; i++) {
    // for (i = save_min_unsignalled; i < save_max_unsignalled; i++) {
        /*
         * signals[i] == 0 indicates we have not received a signal for that
         * vertex yet.
         * signals[i] > 0 indicates the parent vertex + 1 we have received a
         * signal from, but have not handled yet.
         * signals[i] < 0 is the negated (parent vertex + 1) of a vertex we have
         * already handled.
         */
        if (signals[i] > 0) {
            // Mark signals[i] as finished
            
            // fprintf(stderr, "PE %d handling signal %d on local vertex %d with "
            //         "min_unsignalled=%d\n", pe, signals[i], i,
            //         new_min_unsignalled);

            signals[i] *= -1;

            if (i == new_min_unsignalled) {
                /*
                 * Update min unsignalled if we are handling a signal on the
                 * current min unsignalled.
                 */
                new_min_unsignalled++;
            }

            // Handle signal by signalling children
            for (j = local_vertex_offsets[i]; j < local_vertex_offsets[i + 1];
                    j++) {
                const int target_pe = get_owner_pe(neighbors[j], nglobalverts);
                uint64_t min_vertex_on_target = get_starting_vertex_for_pe(
                        target_pe, nglobalverts);

                // int p;
                // for (p = 0; p < nunique_pes; p++) {
                //     if (unique_pes[p] == target_pe) break;
                // }
                // if (p == nunique_pes) {
                //     unique_pes = (int *)realloc(unique_pes, (nunique_pes + 1) *
                //             sizeof(int));
                //     unique_pes[nunique_pes] = target_pe;
                //     nunique_pes++;
                // }

                const int old = shmem_int_cswap(signals + (neighbors[j] -
                            min_vertex_on_target), UNSIGNALLED,
                        local_min_vertex + i + 1, target_pe);

                if (old == UNSIGNALLED) {
                    pe_incrs[target_pe]++;
#ifdef VERBOSE
                    fprintf(stderr, "Signalled %d\n", neighbors[j]);
#endif
                    // Update sent_any_signals
                    sent_any_signals++;
                } else { // Someone else has already signalled
                    nfailed++;
                }
            }
        }

        if (!(signals[i] < 0)) {
            /*
             * There is a race between the signals[i] > 0 check above and other
             * PEs RDMA-ing signals into the signals array. Therefore, if we get
             * to this point we still need to check for a possibly unhandled
             * signal (i.e. signals[i] > 0) even though this conditional looks
             * partially redundant with the one above.
             */
            new_max_unsignalled = i + 1;
        }
    }

    /*
     * To ensure that the updates to each remote PE's signals array arrive
     * before we signal their nsignals.
     */
    shmem_fence();

    for (i = 0; i < npes; i++) {
        if (pe_incrs[i] > 0) {
            shmem_longlong_add((long long int *)nsignals, 2 * pe_incrs[i], i);
        }
    }

    // free(unique_pes);
    free(pe_incrs);

    *min_unsignalled = new_min_unsignalled;
    *max_unsignalled = new_max_unsignalled;

#ifdef VERBOSE
    if (sent_any_signals > 0) {
        fprintf(stderr, "PE %d sent %d signals on iteration %u, nfailed=%lu, "
                "nunique PEs = %lu\n", pe, sent_any_signals, iteration,
                nfailed, nunique_pes);
    }
#endif

    return sent_any_signals;
}

static unsigned count_handled_local_vertices(int *signals,
        uint64_t n_local_vertices) {
    int i;
    unsigned count = 0;
    for (i = 0; i < n_local_vertices; i++) {
        if (signals[i] != 0) {
            count++;
        }
    }
    return count;
}

int main(int argc, char **argv) {
    int num_bfs_roots = 64;
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
    const uint64_t nglobalverts = (uint64_t)(1 << scale);

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
    uint64_t i;
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

    /*
     * Allocate buffers on each PE for storing all edges for which at least one
     * of the vertices of the edge is handled by this PE. This information will
     * be provided by other PEs.
     */
    packed_edge *local_edges = (packed_edge *)shmem_malloc(
            max_n_local_edges * sizeof(packed_edge));
    assert(local_edges);

    int *signals = (int *)shmem_malloc(get_vertices_per_pe(nglobalverts) *
            sizeof(int));
    assert(signals);
    memset(signals, UNSIGNALLED, get_vertices_per_pe(nglobalverts) *
            sizeof(int));

    nsignals = (long long *)shmem_malloc(sizeof(long long));
    assert(nsignals);
    *nsignals = 0;

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

    shmem_free(local_edges);

    // long long expected_signal_count = 0;
    if (get_owner_pe(0, nglobalverts) == pe) {
        // If this PE owns the root vertex, signal it to start traversal.
        signals[0] = 1;
        *nsignals = 2;
    }

#ifdef VERBOSE
    for (i = 0; i < npes; i++) {
        if (i == pe) {
            int j;
            for (j = 0; j < n_local_vertices; j++) {
                unsigned global_vertex_id = local_min_vertex + j;
                fprintf(stderr, "%u:", global_vertex_id);

                int k;
                for (k = local_vertex_offsets[j]; k < local_vertex_offsets[j + 1];
                        k++) {
                    fprintf(stderr, " %u", neighbors[k]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
        shmem_barrier_all();
    }
#endif

    uint64_t min_unsignalled = 0;
    uint64_t max_unsignalled = n_local_vertices;

    int *reduce_dest = (int *)shmem_malloc(sizeof(int));
    int *reduce_src = (int *)shmem_malloc(sizeof(int));

    uint64_t nprocessed = 0;
    uint32_t niterations = 0;
    uint32_t nwaits = 0;
    unsigned long long time_spent_waiting = 0;
    unsigned long long time_spent_signalling = 0;

    int *pWkr_int = (int *)shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(int));
    long *pSync_int = (long *)shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync_int[i] = SHMEM_SYNC_VALUE;
    }

    int *other_signals = (int *)shmem_malloc(npes * sizeof(int));

    shmem_barrier_all();
    const unsigned long long start_bfs = current_time_ns();

    while (1) {
        const unsigned long long start_iter = current_time_ns();

        /*
         * Fences on all signal transmissions ensure that the atomic adds to
         * nsignals must appear after the updates to the signals vector.
         */

        const int sent_any_signals = scan_signals(&min_unsignalled,
                    &max_unsignalled, n_local_vertices, signals,
                    local_vertex_offsets, neighbors,
                    nglobalverts, local_min_vertex, niterations);
        nprocessed += sent_any_signals;

#ifdef VERBOSE
        fprintf(stderr, "PE %d sent %d signals\n", pe, sent_any_signals);
#endif

        const unsigned long long elapsed_iter = current_time_ns() - start_iter;
        time_spent_signalling += elapsed_iter;

        const unsigned long long start_wait = current_time_ns();

        for (i = 0; i < npes; i++) {
            if (i == pe) {
                other_signals[pe] = sent_any_signals;
            } else {
                shmem_int_put_nbi(other_signals + pe, &sent_any_signals,
                        1, i);
            }
        }

        /*
         * Ensure outstanding signals have completed before we check for
         * completion.
         */
        shmem_barrier_all();
        // shmem_quiet();

        int break_out = 1;
        for (i = 0; i < npes; i++) {
            if (other_signals[i] > 0) {
                break_out = 0;
                break;
            }
        }

        const unsigned long long elapsed = current_time_ns() - start_wait;
        time_spent_waiting += elapsed;

        if (break_out) {
            break;
        }

        // *reduce_src = sent_any_signals;
        // shmem_int_max_to_all(reduce_dest, reduce_src, 1, 0, 0, npes, pWrk_int,
        //         pSync_int);
        // if (*reduce_dest == 0) {
        //     break;
        // }

        niterations++;
    }

    shmem_barrier_all();
    const unsigned long long end_bfs = current_time_ns();
    if (pe == 0) {
        fprintf(stderr, "BFS took %f ms\n",
                (double)(end_bfs - start_bfs) / 1000000.0);
    }

    uint64_t count_parents = 0;
    for (i = 0; i < n_local_vertices; i++) {
        if (signals[i] > 0) {
            fprintf(stderr, "Unhandled signal on local vertex %d on PE %d, "
                    "signals[%d] = %d, min_unsignalled=%u, "
                    "max_unsignalled=%u\n", i, pe, i, signals[i],
                    min_unsignalled, max_unsignalled);
            shmem_global_exit(2);
        }
        if (signals[i] < 0) count_parents++;
    }

    fprintf(stderr, "PE %d got out, marked parents for %llu vertices, %f ms "
            "waiting, %f ms signalling, %u iterations, %u waits, %lu parents "
            "found out of %lu local vertices\n", pe,
            nprocessed, (double)time_spent_waiting / 1000000.0,
            (double)time_spent_signalling / 1000000.0,
            niterations, nwaits, count_parents, n_local_vertices);

    shmem_free(signals);

    shmem_finalize();

    return 0;
}
