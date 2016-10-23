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

#define HEADER 0xbbbb

#define QUEUE_SIZE 1048576

#define INCOMING_MAILBOX_SIZE_IN_BYTES 201326592
// #define OUTGOING_MAILBOX_SIZE_IN_BYTES 8388608

#define COALESCING 4096
#define SEND_BUFFER_SIZE (sizeof(int) + sizeof(int) + COALESCING * sizeof(packed_edge))

#define BITS_PER_INT 32

typedef struct _send_buf {
    unsigned char *buf;
    struct _send_buf *next;
} send_buf;

// Save two spots at the start, one for the header and one for the size of the buffer
#define ADD_HEADER(ptr) { \
    *((int *)(ptr)) = HEADER; (ptr) = (void *)(((int *)(ptr)) + 2); \
}

#define GET_SEND_BUF(my_target_pe) { \
    assert(send_bufs[my_target_pe] == NULL); \
    send_buf *gotten = pre_allocated_send_bufs; \
    assert(gotten); \
    pre_allocated_send_bufs = gotten->next; \
    gotten->next = NULL; \
    *((int *)(gotten->buf)) = HEADER; \
    send_bufs[my_target_pe] = gotten; \
    send_bufs_size[my_target_pe] = 2 * sizeof(int); \
}

#define SEND_PACKET(my_target_pe) { \
    assert(send_bufs[my_target_pe]); \
    assert((send_bufs_size[my_target_pe] - (2 * sizeof(int))) % sizeof(packed_edge) == 0); \
    *((int *)(send_bufs[my_target_pe]->buf + sizeof(int))) = (send_bufs_size[my_target_pe] - (2 * sizeof(int))) / sizeof(packed_edge); \
    *nmessages_local += 1; \
\
    fprintf(stderr, "Sending message from %d to %d with size %llu\n", pe, my_target_pe, send_bufs_size[my_target_pe]); \
    const int remote_offset = shmem_int_fadd( \
            recv_buf_index, send_bufs_size[my_target_pe], \
            my_target_pe); \
    shmem_char_put(recv_buf + remote_offset + sizeof(int), \
            send_bufs[my_target_pe]->buf + sizeof(int), \
            send_bufs_size[my_target_pe] - sizeof(int), my_target_pe); \
    shmem_fence(); \
    shmem_char_put(recv_buf + remote_offset, \
            send_bufs[my_target_pe]->buf, sizeof(int), \
            my_target_pe); \
\
    send_bufs[my_target_pe]->next = used_send_bufs; \
    used_send_bufs = send_bufs[my_target_pe]; \
    send_bufs[my_target_pe] = NULL; \
    send_bufs_size[my_target_pe] = 0; \
}

#define SEND_EMPTY_PACKET(my_target_pe) { \
    const int remote_offset = shmem_int_fadd( \
            recv_buf_index, 2 * sizeof(int), \
            my_target_pe); \
    shmem_char_put(recv_buf + remote_offset + sizeof(int), \
            (char *)(empty_packet + 1), sizeof(int), my_target_pe); \
    shmem_fence(); \
    shmem_char_put(recv_buf + remote_offset, (char *)empty_packet, sizeof(int), my_target_pe); \
}

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

static inline void set_visited(const uint64_t vertex, int *visited) {
    const int word_index = vertex / BITS_PER_INT;
    const int bit_index = vertex % BITS_PER_INT;
    const int mask = (1 << bit_index);
    visited[word_index] |= mask;
}

static inline int is_visited(const uint64_t vertex, const int *visited) {
    const int word_index = vertex / BITS_PER_INT;
    const int bit_index = vertex % BITS_PER_INT;
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

    // Logic to remove duplicate edges from neighbors list
    uint64_t total_endpoints = 0;
    uint64_t duplicates = 0;
    for (i = 0; i < n_local_vertices; i++) {
        const int start = local_vertex_offsets[i];
        const int end = local_vertex_offsets[i + 1];
        assert(start <= end);

        const int new_start = start - duplicates;
        assert(new_start >= 0);

        qsort(neighbors + start, end - start, sizeof(*neighbors),
                compare_uint64_t);

        uint64_t curr = neighbors[start];
        int index = start + 1;
        int writing_index = start + 1;
        while (index < end) {
            if (neighbors[index] == curr) {
                // A duplicate of an already seen value
                duplicates++;
                index++;
            } else {
                // Not a duplicate
                curr = neighbors[index];
                neighbors[writing_index] = neighbors[index];
                index++;
                writing_index++;
                total_endpoints++;
            }
        }

        local_vertex_offsets[i] = new_start;
    }
    local_vertex_offsets[n_local_vertices] = total_endpoints;
    neighbors = (uint64_t *)realloc(neighbors, total_endpoints * sizeof(uint64_t));
    assert(neighbors);

#ifdef VERBOSE
    fprintf(stderr, "PE %d found %llu duplicate edges, total endpoints = %llu\n", pe,
            duplicates, total_endpoints);
#endif

    shmem_free(local_edges);

    unsigned char *recv_buf = (unsigned char *)shmem_malloc(INCOMING_MAILBOX_SIZE_IN_BYTES);
    assert(recv_buf);
    unsigned *recv_buf_index = (unsigned *)shmem_malloc(sizeof(unsigned));
    assert(recv_buf_index);
    *recv_buf_index = 0;

    uint64_t *curr_q = (uint64_t *)malloc(QUEUE_SIZE * sizeof(uint64_t));
    assert(curr_q);
    unsigned curr_q_size = 0;
    uint64_t *next_q = (uint64_t *)malloc(QUEUE_SIZE * sizeof(uint64_t));
    assert(next_q);
    unsigned next_q_size = 0;

    short *nmessages_local = (short *)shmem_malloc(sizeof(short));
    short *nmessages_global = (short *)shmem_malloc(sizeof(short));
    assert(nmessages_local && nmessages_global);

    int *empty_packet = (int *)shmem_malloc(2 * sizeof(int));
    assert(empty_packet);
    empty_packet[0] = HEADER;
    empty_packet[1] = 0;

    // Buffers to use to transmit next wave to each other PE, output only
    send_buf *pre_allocated_send_bufs = NULL;
    send_buf *used_send_bufs = NULL;
    unsigned count_pre_allocated = 0;
    while (1) {
        unsigned char *buf = (unsigned char *)shmem_malloc(SEND_BUFFER_SIZE);
        if (!buf) break;
        send_buf *new_send_buf = (send_buf *)malloc(sizeof(send_buf));
        new_send_buf->buf = buf;
        new_send_buf->next = pre_allocated_send_bufs;
        pre_allocated_send_bufs = new_send_buf;
        count_pre_allocated++;
    }

    send_buf **send_bufs = (send_buf **)malloc(npes * sizeof(send_buf *));
    assert(send_bufs);
    unsigned *send_bufs_size = (unsigned *)malloc(npes * sizeof(unsigned));
    assert(send_bufs_size);

    const size_t visited_ints = ((nglobalverts + BITS_PER_INT - 1) /
            BITS_PER_INT);
    const size_t visited_bytes = visited_ints * sizeof(int);
    int *visited = (int *)malloc(visited_bytes);
    assert(visited);

    const unsigned num_bfs_roots = 1;
    assert(num_bfs_roots <= sizeof(bfs_roots) / sizeof(bfs_roots[0]));

    unsigned run;
    for (run = 0; run < num_bfs_roots; run++) {
        memset(visited, 0x00, visited_bytes);
        memset(preds, 0x00, n_local_vertices * sizeof(uint64_t));
        memset(send_bufs, 0x00, npes * sizeof(send_buf *));
        memset(send_bufs_size, 0x00, npes * sizeof(unsigned));

        // uint64_t root = bfs_roots[run];
        uint64_t root = 0;

        set_visited(root, visited);
        if (get_owner_pe(root, nglobalverts) == pe) {
            /*
             * Signal that this PE has received 1 item in its inbox, setting the
             * parent for vertex 0 to 0.
             */
            curr_q[0] = root;
            curr_q_size = 1;
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

            for (i = 0; i < curr_q_size; i++) {
                uint64_t vertex = curr_q[i];

                const int neighbor_start = local_vertex_offsets[vertex - local_min_vertex];
                const int neighbor_end = local_vertex_offsets[vertex - local_min_vertex + 1];

                int j;
                for (j = neighbor_start; j < neighbor_end; j++) {
                    const uint64_t to_explore = neighbors[j];
                    const int target_pe = get_owner_pe(to_explore, nglobalverts);

                    if (!is_visited(to_explore, visited)) {
                        set_visited(to_explore, visited);

                        if (target_pe == pe) {
                            preds[to_explore - local_min_vertex] = vertex;
                            next_q[next_q_size++] = to_explore;
                        } else {
                            if (send_bufs[target_pe] == NULL) {
                                GET_SEND_BUF(target_pe)
                            }

                            unsigned char *write_at =
                                send_bufs[target_pe]->buf +
                                send_bufs_size[target_pe];
                            write_edge((packed_edge *)write_at, to_explore,
                                    vertex);
                            send_bufs_size[target_pe] += sizeof(packed_edge);

                            if (send_bufs_size[target_pe] == SEND_BUFFER_SIZE) {
                                // Send
                                SEND_PACKET(target_pe)
                            }
                        }
                    }
                }
            }

            for (i = 1; i < npes; i++) {
                const int target = (pe + i) % npes;
                if (send_bufs[target]) {
                    // Send any remainders
                    SEND_PACKET(target)
                }
                assert(send_bufs[target] == NULL);

                /*
                 * Send 0-length packet to signify this PE is done sending to
                 * target.
                 */
                SEND_EMPTY_PACKET(target)
            }

            unsigned ndone = 0;
            volatile unsigned char *iter_buf = recv_buf;

            while (ndone < npes - 1) {
                while (*((volatile int *)iter_buf) != HEADER) ;
                *((volatile int *)iter_buf) = 0;
                iter_buf += sizeof(int);

                unsigned buf_size = *((volatile unsigned *)iter_buf);
                iter_buf += sizeof(unsigned);

                if (buf_size == 0) ndone++;

                for (i = 0; i < buf_size; i++) {
                    packed_edge *edge = (packed_edge *)iter_buf;
                    uint64_t to_explore = get_v0_from_edge(edge);
                    uint64_t parent = get_v1_from_edge(edge);

                    if (get_owner_pe(to_explore, nglobalverts) != pe) {
                        fprintf(stderr, "Bad vertex %llu received at pe %d, "
                                "buf_size=%u, parent=%llu\n", to_explore, pe,
                                buf_size, parent);
                        assert(0);
                    }

                    set_visited(parent, visited);
                    if (!is_visited(to_explore, visited)) {
                        preds[to_explore - local_min_vertex] = parent;
                        set_visited(to_explore, visited);
                        next_q[next_q_size++] = to_explore;
                    }

                    iter_buf += sizeof(packed_edge);
                }
            }

            shmem_barrier_all();
            // shmem_quiet();

            *recv_buf_index = 0;

            uint64_t *tmp_q = curr_q;
            curr_q = next_q;
            next_q = tmp_q;

            const unsigned tmp_q_size = curr_q_size;
            curr_q_size = next_q_size;
            next_q_size = tmp_q_size;

            // Restore all send buffers to unused
            while (used_send_bufs) {
                send_buf *save = used_send_bufs->next;
                used_send_bufs->next = pre_allocated_send_bufs;
                pre_allocated_send_bufs = used_send_bufs;
                used_send_bufs = save;
            }

            shmem_short_max_to_all(nmessages_global, nmessages_local, 1, 0, 0,
                    npes, pWrk_short, pSync);

#ifdef PROFILE
            const unsigned long long end_all = current_time_ns();
            reduce_time += (end_all - start_reduce);
#endif

            if (*nmessages_global == 0) {
                break;
            }

            iter++;
        }

        shmem_barrier_all();
        const unsigned long long end_bfs = current_time_ns();
        if (pe == 0) {
            fprintf(stderr, "BFS %d with root=%llu took %f ms, %d iters\n",
                    run, root, (double)(end_bfs - start_bfs) / 1000000.0,
                    iter + 1);
        }

#ifdef VERBOSE
        int count_preds = 0;
        for (i = 0; i < n_local_vertices; i++) {
            if (preds[i] > 0) count_preds++;
        }
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