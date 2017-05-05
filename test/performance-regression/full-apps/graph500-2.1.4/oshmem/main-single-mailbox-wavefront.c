#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <shmem.h>
#include <shmemx.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>

#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <limits.h>
#include <pthread.h>

// #define VERBOSE

// #ifdef USE_CRC
// #include "crc.h"
// typedef int32_t size_type;
// #elif USE_MURMUR
// #include "MurmurHash3.h"
// typedef uint32_t crc;
// typedef int32_t size_type;
// #elif USE_CITY32
// #include "city.h"
// typedef uint32_t crc;
// typedef int32_t size_type;
// #elif USE_CITY64
// #include "city.h"
// typedef uint64_t crc;
// typedef int64_t size_type;
// #else
// #error No hashing algorithm specific
// #endif

#include "mrg.h"
#include "packed_edge.h"
#include "utilities.h"
#include "generator.h"

// #define QUEUE_SIZE 1572864
#define QUEUE_SIZE 1048576

// #define INCOMING_MAILBOX_SIZE_IN_BYTES 100663296
#define INCOMING_MAILBOX_SIZE_IN_BYTES (200 * 1024 * 1024)

/*
 * Header format:
 *
 *   sizeof(crc) bytes                                       : header checksum
 *   sizeof(size_type) bytes                                 : Length of whole packet in bytes (N)
 *   sizeof(crc) bytes                                       : CRC32 body checksum
 *   N - sizeof(crc) - sizeof(size_type) - sizeof(crc) bytes : Body
 */
#define COALESCING 512
#define SEND_HEADER_SIZE (sizeof(crc) + sizeof(size_type) + sizeof(crc))
#define SEND_BUFFER_SIZE (SEND_HEADER_SIZE + COALESCING * sizeof(packed_edge))

#define BITS_PER_BYTE 8
#define BITS_PER_INT (sizeof(unsigned) * BITS_PER_BYTE)

#define MAX_ITERS 10

typedef struct {
    short steady_state;
    short bufs[2];
} vert_info;

typedef struct _send_buf {
    unsigned char *buf;
    struct _send_buf *next;
} send_buf;

#define SEND_BUF_SIZE_TO_NEDGES(my_send_buf_size) (((my_send_buf_size) - SEND_HEADER_SIZE) / sizeof(packed_edge))

#define GET_SEND_BUF(my_target_pe) { \
    assert(send_bufs[my_target_pe] == NULL); \
    send_buf *gotten = pre_allocated_send_bufs; \
    assert(gotten); \
    pre_allocated_send_bufs = gotten->next; \
    send_bufs[my_target_pe] = gotten; \
    send_bufs_size[my_target_pe] = SEND_HEADER_SIZE; \
}

#define PREPARE_PACKET(my_target_pe) { \
    assert(send_bufs[my_target_pe]); \
    const unsigned send_buf_size = send_bufs_size[my_target_pe]; \
    assert((send_buf_size - SEND_HEADER_SIZE) % sizeof(packed_edge) == 0); \
    assert(send_buf_size <= SEND_BUFFER_SIZE); \
    const unsigned nedges = SEND_BUF_SIZE_TO_NEDGES(send_buf_size); \
    unsigned char *send_buf = send_bufs[my_target_pe]->buf; \
    /* Save the total size of this packet */ \
    *((size_type *)(send_buf + sizeof(crc))) = send_buf_size; \
    /* Save the CRC of the body of this packet */ \
    *((crc *)(send_buf + sizeof(crc) + sizeof(size_type))) = hash( \
            (const unsigned char *)(send_buf + SEND_HEADER_SIZE), \
            send_buf_size - SEND_HEADER_SIZE); \
    /* Save the CRC of the header of this packet */ \
    *((crc *)send_buf) = hash( \
            (const unsigned char *)(send_buf + sizeof(crc)), \
            SEND_HEADER_SIZE - sizeof(crc)); \
}

#define SEND_PACKET(my_target_pe) { \
    PREPARE_PACKET(my_target_pe) \
\
    const int remote_offset = shmem_int_fadd( \
            recv_buf_index, send_bufs_size[my_target_pe], \
            my_target_pe); \
    assert(remote_offset + send_bufs_size[my_target_pe] < INCOMING_MAILBOX_SIZE_IN_BYTES); \
    shmem_char_put_nbi((char *)(recv_buf + remote_offset), \
            (const char *)send_bufs[my_target_pe]->buf, \
            send_bufs_size[my_target_pe], my_target_pe); \
\
    send_bufs[my_target_pe] = NULL; \
    send_bufs_size[my_target_pe] = 0; \
}

// #define VERBOSE
// #define PROFILE

static int pe = -1;
static int npes = -1;

void sig_handler(int signo) {
    fprintf(stderr, "%d: received signal %d %d\n", pe, signo, SIGUSR1);

    raise(SIGABRT);
    assert(0); // should never reach here
}

void *kill_func(void *data) {
    int kill_seconds = *((int *)data);
    int err = sleep(kill_seconds);
    assert(err == 0);
    fprintf(stderr, "hitting pe %d with SUGUSR1\n", pe);
    raise(SIGUSR1);
    return NULL;
}

#ifdef PROFILE
unsigned long long hash_time = 0;
unsigned long long hash_calls = 0;
unsigned long long wasted_hashes = 0;
unsigned long long total_packets_received = 0;
unsigned long long n_packets_wasted = 0;
unsigned long long total_elements_received = 0;
unsigned long long n_elements_wasted = 0;
#ifdef DETAILED_PROFILE
unsigned *wavefront_visited = NULL;
unsigned long long duplicates_in_same_wavefront = 0;
unsigned long long duplicates_in_same_wavefront_total = 0;
#endif
#endif

// static inline crc hash(const unsigned char * const data, const size_t len) {
// #ifdef PROFILE
//     const unsigned long long start_time = current_time_ns();
// #endif
// 
//     crc result;
// #ifdef USE_CRC
//     result = crcFast(data, len);
// #elif USE_MURMUR
//     MurmurHash3_x86_32(data, len, 12345, &result);
// #elif USE_CITY32
//     result = CityHash32((const char *)data, len);
// #elif USE_CITY64
//     result = CityHash64((const char *)data, len);
// #else
// #error No hashing algorithm specified
// #endif
// 
// #ifdef PROFILE
//     hash_time += (current_time_ns() - start_time);
//     hash_calls++;
// #endif
// 
//     return result;
// }

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
    const unsigned word_index = global_vertex_id / BITS_PER_INT;
    assert(word_index < visited_length);
    const int bit_index = global_vertex_id % BITS_PER_INT;
    const int mask = (1 << bit_index);

    return (((visited[word_index] & mask) > 0) ? 1 : 0);
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

static inline void loop_body(vert_info *verts, const uint64_t local_vertex_id,
        const int last_put_index, const unsigned *local_vertex_offsets,
        const uint64_t *neighbors, const uint64_t nglobalverts,
        unsigned *shared_visited, const size_t visited_ints,
        const uint64_t local_min_vertex, unsigned *local_visited,
        const int updating_index, const short *pe_plus_one_ptr,
        shmemx_ctx_t *curr_ctx, shmemx_ctx_t *idle_ctx, int *thread_natomics,
        unsigned long long *time_blocked_on_quiet) {
    const short curr = verts[local_vertex_id].bufs[last_put_index];
    // const short curr = last_put[local_vertex_id];

    // if (curr > 0 && steady_state[local_vertex_id] == 0)
    if (curr > 0 && verts[local_vertex_id].steady_state == 0)
    {
        const int neighbor_start = local_vertex_offsets[local_vertex_id];
        const int neighbor_end = local_vertex_offsets[local_vertex_id + 1];

        int j;
        for (j = neighbor_start; j < neighbor_end; j++) {
            const uint64_t to_explore_global_id = neighbors[j];
            const int target_pe = get_owner_pe(to_explore_global_id, nglobalverts);
            const uint64_t to_explore_local_id = to_explore_global_id -
                get_starting_vertex_for_pe(target_pe, nglobalverts);
            assert(to_explore_local_id < get_vertices_per_pe(nglobalverts));

            const int already_visited = is_visited(
                    to_explore_global_id, shared_visited,
                    visited_ints, local_min_vertex) ||
                is_visited(to_explore_global_id, local_visited,
                        visited_ints, local_min_vertex);

            if (!already_visited) {
                shmemx_ctx_putmem(
                        &(verts[to_explore_local_id].bufs[updating_index]),
                        pe_plus_one_ptr, sizeof(*pe_plus_one_ptr), target_pe, *curr_ctx);
                // shmemx_ctx_short_put_nbi(
                //         &(verts[to_explore_local_id].bufs[updating_index]),
                //         pe_plus_one_ptr, 1, target_pe, *curr_ctx);
                // shmemx_ctx_short_put_nbi(updating + to_explore_local_id,
                //         &pe_plus_one, 1, target_pe, curr_ctx);
                // shmem_int_put(updating + to_explore_local_id,
                //         &pe_plus_one, 1, target_pe);
                // shmemx_ctx_short_p(updating + to_explore_local_id,
                //         pe_plus_one, target_pe, curr_ctx);
                // shmem_short_p(updating + to_explore_local_id,
                //         pe_plus_one, target_pe);
                int curr_n_atomics = *thread_natomics + 1;
                *thread_natomics = curr_n_atomics;

                if (curr_n_atomics % 256 == 0) {
                    const unsigned long long start_quiet = current_time_ns();
                    shmemx_ctx_quiet(*idle_ctx);
                    const unsigned long long elapsed_quiet = current_time_ns() - start_quiet;
                    *time_blocked_on_quiet += elapsed_quiet;

                    shmemx_ctx_t tmp_ctx = *curr_ctx;
                    *curr_ctx = *idle_ctx;
                    *idle_ctx = *curr_ctx;
                }

                set_visited(to_explore_global_id, local_visited,
                        visited_ints, local_min_vertex);
                if (target_pe == pe) {
                    set_visited(to_explore_global_id, shared_visited,
                            visited_ints, local_min_vertex);
                }
            }
        }

        verts[local_vertex_id].steady_state = curr;
        verts[local_vertex_id].bufs[last_put_index] = 0;
        // steady_state[local_vertex_id] = curr;
        // last_put[local_vertex_id] = 0;
        set_visited(local_min_vertex + local_vertex_id,
                shared_visited,visited_ints, local_min_vertex);
    }
}

int main(int argc, char **argv) {
#ifdef USE_CRC
    crcInit();
#endif

    if (argc < 4) {
        fprintf(stderr, "usage: %s scale edgefactor num-bfs-roots\n",
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
    const int num_bfs_roots = atoi(argv[3]);

    // __sighandler_t serr = signal(SIGUSR1, sig_handler);
    // assert(serr != SIG_ERR);

    // int kill_seconds = 120;
    // pthread_t thread;
    // const int perr = pthread_create(&thread, NULL, kill_func,
    //         (void *)&kill_seconds);
    // assert(perr == 0);

    int provided;
    shmemx_init_thread(SHMEMX_THREAD_MULTIPLE, &provided);
    assert(provided == SHMEMX_THREAD_MULTIPLE);

    int nthreads;
#pragma omp parallel
#pragma omp master
    {
        nthreads = omp_get_num_threads();
    }

    uint64_t i;
    shmemx_domain_t *domains = (shmemx_domain_t *)malloc(nthreads * sizeof(*domains));
    shmemx_ctx_t *contexts = (shmemx_ctx_t *)malloc(2 * nthreads * sizeof(*contexts));
    assert(domains && contexts);

    int err = shmemx_domain_create(SHMEMX_THREAD_SINGLE,
            nthreads, domains);
    assert(err == 0); 

    for (i = 0; i < nthreads; i++) {
        err = shmemx_ctx_create(domains[i], contexts + (2 * i));
        assert(err == 0);
        err = shmemx_ctx_create(domains[i], contexts + (2 * i + 1));
        assert(err == 0);
    }

    pe = shmem_my_pe();
    npes = shmem_n_pes();

    uint_fast32_t seed[5];
    uint64_t seed1 = 2, seed2 = 3;
    make_mrg_seed(seed1, seed2, seed);

    const uint64_t edges_per_pe = (nglobaledges + npes - 1) / npes;
    const uint64_t start_edge_index = pe * edges_per_pe;
    int64_t nedges_this_pe = edges_per_pe;
    if (start_edge_index + nedges_this_pe > nglobaledges) {
        nedges_this_pe = nglobaledges - start_edge_index;
        if (nedges_this_pe < 0) nedges_this_pe = 0;
    }

    if (pe == 0) {
        fprintf(stderr, "%llu: %lu total vertices, %lu total edges, %d PEs, "
                "~%lu edges per PE, ~%lu vertices per PE, %d threads per PE\n",
                current_time_ns(), nglobalverts, nglobaledges, npes,
                edges_per_pe, get_vertices_per_pe(nglobalverts), nthreads);
    }

    /*
     * Use the Graph500 utilities to generate a set of edges distributed across
     * PEs.
     */
#ifdef VERBOSE
    fprintf(stderr, "PE %d malloc-ing %llu bytes\n", shmem_my_pe(),
            nedges_this_pe * sizeof(packed_edge));
#endif
    packed_edge *actual_buf = (packed_edge *)malloc(
            nedges_this_pe * sizeof(packed_edge));
    assert(actual_buf || nedges_this_pe == 0);
    generate_kronecker_range(seed, scale, start_edge_index,
            start_edge_index + nedges_this_pe, actual_buf);

    /*
     * Count the number of edge endpoints in actual_buf that are resident on
     * each PE.
     */
#ifdef VERBOSE
    fprintf(stderr, "PE %d calloc-ing %llu bytes\n", shmem_my_pe(),
            npes * sizeof(long long));
#endif
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
#ifdef VERBOSE
    fprintf(stderr, "PE %d malloc-ing %llu bytes\n", shmem_my_pe(),
            npes * sizeof(long long));
#endif
    long long *remote_offsets = (long long *)malloc(npes * sizeof(long long));
    assert(remote_offsets);
    for (i = 0; i < npes; i++) {
        remote_offsets[i] = shmem_longlong_fadd((long long int *)&n_local_edges,
                count_edges_shared_with_pe[i], i);
    }
    free(count_edges_shared_with_pe);

    shmem_barrier_all();

#ifdef VERBOSE
    fprintf(stderr, "PE %d shmem_malloc-ing %llu bytes\n", shmem_my_pe(),
            SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
#endif
    int *pWrkInt = (int *)shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*pWrkInt));
    long long *pWrkLongLong = (long long *)shmem_malloc(
            SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(*pWrkLongLong));
    assert(pWrkInt && pWrkLongLong);

    long *pSync = (long *)shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
#ifdef VERBOSE
    fprintf(stderr, "PE %d shmem_malloc-ing %llu bytes\n", shmem_my_pe(),
            SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
#endif
    long *pSync2 = (long *)shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    assert(pSync && pSync2);
    for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
        pSync2[i] = SHMEM_SYNC_VALUE;
    }
    shmem_longlong_max_to_all((long long int *)&max_n_local_edges,
            (long long int *)&n_local_edges, 1, 0, 0, npes, pWrkLongLong, pSync);

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
#ifdef VERBOSE
    fprintf(stderr, "PE %d shmem_malloc-ing %llu bytes\n", shmem_my_pe(),
            max_n_local_edges * sizeof(packed_edge));
#endif
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
        shmem_quiet();
    }

    free(remote_offsets);

    shmem_barrier_all();

    free(actual_buf);

#ifdef VERBOSE
    fprintf(stderr, "PE %d calloc-ing %llu bytes\n", shmem_my_pe(),
            (n_local_vertices + 1) * sizeof(unsigned));
#endif
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
#ifdef VERBOSE
    fprintf(stderr, "PE %d malloc-ing %llu bytes\n", shmem_my_pe(),
            acc * 2 * sizeof(uint64_t));
#endif
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

    // Remove duplicate edges in neighbors
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
#ifdef VERBOSE
    fprintf(stderr, "PE %d realloc-ing from %llu bytes to %llu bytes\n", shmem_my_pe(),
            acc * 2 * sizeof(uint64_t), writing_index * sizeof(uint64_t));
#endif
    neighbors = (uint64_t *)realloc(neighbors, writing_index *
            sizeof(uint64_t));
    assert(writing_index == 0 || neighbors);

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

    // For debugging, print all vertices
    // {
    //     int k;
    //     for (k = 0; k < npes; k++) {
    //         if (k == shmem_my_pe()) {
    //             for (i = 0; i < n_local_vertices; i++) {
    //                 const unsigned neighbors_start = local_vertex_offsets[i];
    //                 const unsigned neighbors_end = local_vertex_offsets[i + 1];

    //                 fprintf(stderr, "HOWDY %d :", local_min_vertex + i);
    //                 int j;
    //                 for (j = neighbors_start; j < neighbors_end; j++) {
    //                     fprintf(stderr, " %d", neighbors[j]);
    //                 }
    //                 fprintf(stderr, "\n");
    //             }

    //         }
    //         shmem_barrier_all();
    //     }
    // }

    shmem_free(local_edges);

    int *first_traversed_by = (int *)shmem_malloc(
            get_vertices_per_pe(nglobalverts) * sizeof(*first_traversed_by));
    assert(first_traversed_by);

    int *my_n_signalled = (int *)shmem_malloc(sizeof(*my_n_signalled));
    assert(my_n_signalled);
    int *total_n_signalled = (int *)shmem_malloc(sizeof(*total_n_signalled));
    assert(total_n_signalled);
    int *my_natomics = (int *)shmem_malloc(sizeof(*my_natomics));
    assert(my_natomics);
    int *total_natomics = (int *)shmem_malloc(sizeof(*total_natomics));
    assert(total_natomics);

    const size_t visited_ints = ((nglobalverts + BITS_PER_INT - 1) /
            BITS_PER_INT);
    const size_t visited_bytes = visited_ints * sizeof(unsigned);
    unsigned *shared_visited = (unsigned *)shmem_malloc(visited_bytes);
    assert(shared_visited);
    unsigned *local_visited = (unsigned *)malloc(visited_bytes);
    assert(local_visited);

    vert_info *verts = (vert_info *)shmem_malloc(get_vertices_per_pe(nglobalverts) *
            sizeof(*verts));
    assert(verts);
    // short *steady_state = (short *)shmem_malloc(get_vertices_per_pe(nglobalverts) * sizeof(*steady_state));
    // assert(steady_state);
    // short *last_put = (short *)shmem_malloc(get_vertices_per_pe(nglobalverts) * sizeof(*last_put));
    // assert(last_put);
    // short *updating = (short *)shmem_malloc(get_vertices_per_pe(nglobalverts) * sizeof(*updating));
    // assert(updating);

    int old;

    unsigned run;
    for (run = 0; run < num_bfs_roots; run++) {

        memset(first_traversed_by, 0x00,
                get_vertices_per_pe(nglobalverts) * sizeof(*first_traversed_by));
        memset(shared_visited, 0x00, visited_bytes);
        memset(local_visited, 0x00, visited_bytes);
        memset(verts, 0x00, get_vertices_per_pe(nglobalverts) * sizeof(*verts));
        // memset(steady_state, 0x00, get_vertices_per_pe(nglobalverts) * sizeof(*steady_state));
        // memset(last_put, 0x00, get_vertices_per_pe(nglobalverts) * sizeof(*last_put));
        // memset(updating, 0x00, get_vertices_per_pe(nglobalverts) * sizeof(*updating));

        int last_put_index = 0;
        uint64_t root = 0;
        set_visited(root, shared_visited, visited_ints, local_min_vertex);
        set_visited(root, local_visited, visited_ints, local_min_vertex);

        if (get_owner_pe(root, nglobalverts) == pe) {
            // last_put[root - local_min_vertex] = pe + 1;
            verts[root - local_min_vertex].bufs[last_put_index] = pe + 1;
        }

        shmem_barrier_all();
        const unsigned long long start_bfs = current_time_ns();
        int iter = 0;
        *my_natomics = 0;

        int prev_prev_put = 0;
        int prev_put = 0;

        do {
            *my_n_signalled = 0;
            // memset(my_n_signalled, 0x00, (npes + 1) * sizeof(*my_n_signalled));
            int old; // unused
            short pe_plus_one = pe + 1;
            const int updating_index = (last_put_index + 1) % 2;

            int reduced_natomics = 0;
            unsigned long long reduced_quiet_time = 0;

            const int put_delta = prev_put - prev_prev_put;

            const unsigned long long start_iter = current_time_ns();
#pragma omp parallel reduction(+:reduced_natomics) reduction(+:reduced_quiet_time) default(none) \
            firstprivate(pe_plus_one, my_n_signalled, pe, \
                    npes, local_min_vertex, local_max_vertex, verts, \
                    local_visited, neighbors, \
                    shared_visited, local_vertex_offsets, n_local_vertices, contexts, last_put_index, put_delta)
            {

                shmemx_ctx_t curr_ctx = contexts[2 * omp_get_thread_num()];
                shmemx_ctx_t idle_ctx = contexts[2 * omp_get_thread_num() + 1];
                int thread_natomics = 0;
                unsigned long long thread_quiet_time = 0;

                uint64_t local_vertex_id;
                const int nchunks = 4 * omp_get_num_threads();
                const int chunk_size = (n_local_vertices + nchunks - 1) /
                    nchunks;

                if (put_delta > 100000) {
#pragma omp for schedule(dynamic,chunk_size)
                    for (local_vertex_id = 0;
                            local_vertex_id < n_local_vertices; local_vertex_id++) {

                        loop_body(verts, local_vertex_id, last_put_index,
                                local_vertex_offsets, neighbors, nglobalverts,
                                shared_visited, visited_ints, local_min_vertex,
                                local_visited, updating_index, &pe_plus_one,
                                &curr_ctx, &idle_ctx, &thread_natomics,
                                &thread_quiet_time);
                    }
                } else {
#pragma omp for schedule(static)
                    for (local_vertex_id = 0;
                            local_vertex_id < n_local_vertices; local_vertex_id++) {

                        loop_body(verts, local_vertex_id, last_put_index,
                                local_vertex_offsets, neighbors, nglobalverts,
                                shared_visited, visited_ints, local_min_vertex,
                                local_visited, updating_index, &pe_plus_one,
                                &curr_ctx, &idle_ctx, &thread_natomics,
                                &thread_quiet_time);
                    }
                }

                const size_t min_byte_to_send = local_min_vertex / BITS_PER_BYTE;
                const size_t max_byte_to_send = local_max_vertex / BITS_PER_BYTE;
                const size_t bytes_to_send = max_byte_to_send - min_byte_to_send + 1;
                char *my_visited = ((char *)shared_visited) + min_byte_to_send;

#pragma omp for schedule(static)
                for (i = 0; i < npes - 1; i++) {
                    if (i >= pe) {
                        i = i + 1;
                    }
                    shmemx_ctx_putmem_nbi(my_visited, my_visited, bytes_to_send, i,
                            curr_ctx);
                }

                reduced_natomics += thread_natomics;
                reduced_quiet_time += thread_quiet_time;
            }

            const unsigned long long end_kernel = current_time_ns();

            *my_natomics += reduced_natomics;
            *my_n_signalled = reduced_natomics;

            shmem_int_sum_to_all(total_n_signalled, my_n_signalled, 1, 0, 0,
                    npes, pWrkInt, pSync);

            const unsigned long long end_reduction = current_time_ns();

            for (i = 0; i < 2 * nthreads; i++) {
                shmemx_ctx_quiet(contexts[i]);
            }
            shmem_barrier_all();

            const unsigned long long end_iter = current_time_ns();
            printf("PE %d iter %d kernel = %f ms, reduce = %f ms, barrier = %f "
                    "ms, %d sends / %llu vertices, signalled = %d, %s, "
                    "quiet time = %f ms\n", pe, iter,
                    (double)(end_kernel - start_iter) / 1000000.0,
                    (double)(end_reduction - end_kernel) / 1000000.0,
                    (double)(end_iter - end_reduction) / 1000000.0,
                    reduced_natomics, nglobalverts, *total_n_signalled,
                    (put_delta > 100000 ? "dynamic" : "static"),
                    (double)reduced_quiet_time / 1000000.0);

            // short *tmp = last_put;
            // last_put = updating;
            // updating = tmp;

            iter++;
            last_put_index = (last_put_index + 1) % 2;

            prev_prev_put = prev_put;
            prev_put = *total_n_signalled;
        } while (*total_n_signalled > 0);

        const unsigned long long end_bfs = current_time_ns();

        // For debugging on small datasets, print results
        // for (i = 0; i < nglobalverts; i++) {
        //     if (get_owner_pe(i, nglobalverts) == pe) {
        //         printf("Vertex %d : traversed by %d\n", i,
        //                 first_traversed_by[i - local_min_vertex]);
        //     }
        //     shmem_barrier_all();
        // }

        // Lightweight validation
        // int count_not_set = 0;
        // int count_not_handled = 0;
        // int count_set = 0;
        // for (i = 0; i < n_local_vertices; i++) {
        //     const int curr = first_traversed_by[i];
        //     if (curr == 0) {
        //         count_not_set++;
        //     } else if (curr > 0) {
        //         count_not_handled++;
        //     } else {
        //         count_set++;
        //     }
        // }
        // fprintf(stderr, "PE %d, run %d : set %d , not set %d , unhandled %d\n",
        //         shmem_my_pe(), run, count_set, count_not_set, count_not_handled);

        shmem_int_sum_to_all(total_natomics, my_natomics, 1, 0, 0,
                npes, pWrkInt, pSync);
        shmem_barrier_all();

        if (pe == 0) {
            fprintf(stderr, "BFS %d with root=%llu took %f ms, %d iters, %d atomics\n",
                    run, root, (double)(end_bfs - start_bfs) / 1000000.0,
                    iter, *total_natomics);
        }
    }

    for (i = 0; i < nthreads; i++) {
        shmemx_ctx_destroy(contexts[i]);
    }
    shmemx_domain_destroy(nthreads, domains);

    shmem_finalize();

    return 0;
}
