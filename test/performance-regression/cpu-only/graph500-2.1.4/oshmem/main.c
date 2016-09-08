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

#define UNSIGNALLED 0

volatile long long *nsignals = NULL;
int ndone = 0;
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

static int scan_signals(uint64_t n_local_vertices, int *signals,
        unsigned *local_vertex_offsets, uint64_t *neighbors,
        uint64_t nglobalverts, const uint64_t local_min_vertex,
        const uint32_t iteration) {
    uint64_t i, j;

    int *unique_verts = NULL;
    uint64_t nunique_verts = 0;
    int *unique_pes = NULL;
    uint64_t nunique_pes = 0;
    uint64_t nfailed = 0;

    long long int *pe_incrs = (long long int *)malloc(npes * sizeof(long long int));
    memset(pe_incrs, 0x00, npes * sizeof(long long int));

    int sent_any_signals = 0;
    for (i = 0; i < n_local_vertices; i++) {
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
            signals[i] *= -1;

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
                    pe_incrs[target_pe] += 1;
                    // Update sent_any_signals
                    sent_any_signals++;
                } else {
                    nfailed++;
                }
            }
        }
    }
    shmem_fence();
    for (i = 0; i < npes; i++) {
        if (pe_incrs[i] > 0) {
            shmem_longlong_add((long long int *)nsignals, pe_incrs[i], i);
        }
    }

    free(unique_pes);
    free(pe_incrs);


    fprintf(stderr, "PE %d sent %d signals on iteration %u, nfailed=%lu, "
            "nunique PEs = %lu\n", pe, sent_any_signals, iteration,
            nfailed, nunique_pes);

    return sent_any_signals;
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
        fprintf(stderr, "%lu total vertices, %lu total edges, %d PEs, ~%lu edges per "
                "PE, ~%lu vertices per PE\n", nglobalverts, nglobaledges, npes,
                edges_per_pe, get_vertices_per_pe(nglobalverts));
    }
    packed_edge *actual_buf = (packed_edge *)malloc(
            nedges_this_pe * sizeof(packed_edge));
    assert(actual_buf || nedges_this_pe == 0);
    generate_kronecker_range(seed, scale, start_edge_index,
            start_edge_index + nedges_this_pe, actual_buf);

    long long *local_incrs = (long long *)malloc(npes * sizeof(long long));
    assert(local_incrs);
    memset(local_incrs, 0x00, npes * sizeof(long long));
    uint64_t i;
    for (i = 0; i < nedges_this_pe; i++) {
        int64_t v0 = get_v0_from_edge(actual_buf + i);
        int64_t v1 = get_v1_from_edge(actual_buf + i);
        int v0_pe = get_owner_pe(v0, nglobalverts);
        int v1_pe = get_owner_pe(v1, nglobalverts);
        local_incrs[v0_pe] += 1;
        local_incrs[v1_pe] += 1;
    }

    long long *remote_offsets = (long long *)malloc(npes * sizeof(long long));
    assert(remote_offsets);
    for (i = 0; i < npes; i++) {
        remote_offsets[i] = shmem_longlong_fadd((long long int *)&n_local_edges,
                local_incrs[i], i);
    }
    free(local_incrs);

    fprintf(stderr, "PE %d > A\n", pe);

    shmem_barrier_all();

    fprintf(stderr, "PE %d > B\n", pe);

    for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
    shmem_longlong_max_to_all((long long int *)&max_n_local_edges,
            (long long int *)&n_local_edges, 1, 0, 0, npes, pWrk, pSync);

    fprintf(stderr, "PE %d > C\n", pe);

    if (pe == 0) {
        fprintf(stderr, "Max. # local edges = %lld\n", max_n_local_edges);
    }

    uint64_t local_min_vertex = get_starting_vertex_for_pe(pe, nglobalverts);
    uint64_t local_max_vertex = get_ending_vertex_for_pe(pe, nglobalverts);
    uint64_t n_local_vertices;
    if (local_min_vertex >= local_max_vertex) {
        n_local_vertices = 0;
    } else {
        n_local_vertices = local_max_vertex - local_min_vertex;
    }

    packed_edge *local_edges = (packed_edge *)shmalloc(
            max_n_local_edges * sizeof(packed_edge));
    assert(local_edges);
    int *signals = (int *)shmalloc(get_vertices_per_pe(nglobalverts) *
            sizeof(int));
    assert(signals);
    memset(signals, UNSIGNALLED, get_vertices_per_pe(nglobalverts) *
            sizeof(int));
    nsignals = (long long *)shmalloc(sizeof(long long));
    assert(nsignals);

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
    fprintf(stderr, "PE %d > D\n", pe);

    shmem_barrier_all();

    fprintf(stderr, "PE %d > E\n", pe);

    free(actual_buf);

    unsigned *local_vertex_offsets = (unsigned *)malloc(
            (n_local_vertices + 1) * sizeof(unsigned));
    assert(local_vertex_offsets);
    memset(local_vertex_offsets, 0x00, n_local_vertices * sizeof(unsigned));

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

    uint64_t acc = 0;
    for (i = 0; i < n_local_vertices; i++) {
        uint64_t new_acc = acc + local_vertex_offsets[i];
        local_vertex_offsets[i] = new_acc; // point to the last element
        acc = new_acc;
    }
    local_vertex_offsets[n_local_vertices] = acc;

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

    shfree(local_edges);

    long long expected_signal_count = 0;
    if (get_owner_pe(0, nglobalverts) == pe) {
        signals[0] = 1;
        *nsignals++;
    }

    uint64_t nprocessed = 0;
    uint32_t niterations = 0;
    uint32_t checked_done = 0;
    unsigned long long time_spent_waiting = 0;
    unsigned long long time_spent_in_barriers = 0;
    shmem_barrier_all();
    const unsigned long long start_bfs = current_time_ns();

    while (1) {
        const int sent_any_signals = scan_signals(n_local_vertices, signals,
                local_vertex_offsets, neighbors, nglobalverts, local_min_vertex,
                niterations);
        nprocessed += sent_any_signals;

        if (checking_if_done) {
            shmem_int_add(&nstill_working, sent_any_signals, 0);
            checking_if_done = 0;
            const unsigned long long start_barrier = current_time_ns();
            shmem_barrier_all();
            shmem_barrier_all();
            time_spent_in_barriers += (current_time_ns() - start_barrier);
            if (shmem_int_fadd(&nstill_working, 0, 0) == -1) {
                // All done
                break;
            }
        }

        const int ncomplete = shmem_int_finc(&ndone, 0) + 1;
        if (ncomplete == npes) {
            // Signal all and make sure all are complete
            checked_done += 1;
            fprintf(stderr, "PE %d signalling to check if done after %f ms\n",
                    pe, (current_time_ns() - start_bfs) / 1000000.0);
            for (i = 0; i < npes; i++) {
                int one = 1;
                if (i != pe) {
                    shmem_int_put(&checking_if_done, &one, 1, i);
                }
            }

            shmem_fence();

            // Signal all to wake up and check if done
            for (i = 0; i < npes; i++) {
                if (i != pe) {
                    shmem_longlong_inc((long long int *)nsignals, i);
                }
            }

            const int sent_any_signals = scan_signals(n_local_vertices, signals,
                    local_vertex_offsets, neighbors, nglobalverts,
                    local_min_vertex, niterations);
            nprocessed += sent_any_signals;
            shmem_int_add(&nstill_working, sent_any_signals, 0);

            unsigned long long start_barrier = current_time_ns();
            shmem_barrier_all();
            time_spent_in_barriers += (current_time_ns() - start_barrier);

            const int any_working = shmem_int_fadd(&nstill_working, 0, 0);
            if (any_working) {
                shmem_int_swap(&nstill_working, 0, 0);
            } else {
                shmem_int_swap(&nstill_working, -1, 0);
                break;
            }
            expected_signal_count = *nsignals;
            shmem_int_add(&ndone, -1, 0);

            start_barrier = current_time_ns();
            shmem_barrier_all();
            time_spent_in_barriers += (current_time_ns() - start_barrier);
        } else {
            const unsigned long long start_wait = current_time_ns();
            shmem_longlong_wait_until((long long int *)nsignals, SHMEM_CMP_GT,
                    expected_signal_count);
            time_spent_waiting += (current_time_ns() - start_wait);

            fprintf(stderr, "PE %d bumping from %lld to %lld\n",
                    pe, expected_signal_count, *nsignals);
            expected_signal_count = *nsignals;
            shmem_int_add(&ndone, -1, 0);
        }
        niterations++;
    }

    shmem_barrier_all();
    const unsigned long long end_bfs = current_time_ns();
    if (pe == 0) {
        fprintf(stderr, "BFS took %f ms\n", (double)(end_bfs - start_bfs) / 1000000.0);
    }

    if (expected_signal_count != *nsignals) {
        fprintf(stderr, "PE %d expected # signals to be %lld but was %lld\n",
                expected_signal_count, *nsignals);
        exit(1);
    }

    uint64_t count_parents = 0;
    for (i = 0; i < n_local_vertices; i++) {
        assert(signals[i] <= 0);
        if (signals[i] < 0) count_parents++;
    }

    fprintf(stderr, "PE %d got out, processed %llu vertices, %f ms waiting, "
            "%f ms in barriers, %u iterations, %u checked done, %lu parents "
            "found out of %lu local vertices\n", pe,
            nprocessed, (double)time_spent_waiting / 1000000.0,
            (double)time_spent_in_barriers / 1000000.0, niterations,
            checked_done, count_parents, n_local_vertices);

    shfree(signals);

    shmem_finalize();

    return 0;
}
