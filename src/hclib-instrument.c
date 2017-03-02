// #define _POSIX_C_SOURCE 1
// #define __USE_POSIX199309 1
#define _GNU_SOURCE

#include <signal.h>
#include <errno.h>
#include <aio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "hclib.h"
#include "hclib-instrument.h"

#define EVENT_BUFFER_LENGTH 2048

static hclib_event_type_info *event_types = NULL;
static unsigned n_event_types = 0;

static hclib_instrument_event **active_thread_buffers = NULL;
static hclib_instrument_event **flushing_thread_buffers = NULL;
static unsigned *n_buffered_events = NULL;
struct aiocb flushing_cb;

static unsigned save_nthreads = 0;

static FILE **dump_files = NULL;
static off_t *dump_file_offsets = NULL;

static unsigned *thread_event_counters = NULL;

static char *dump_folder = NULL;

static void flush_events(const int tid) {

    FILE *fp = dump_files[tid];
    const int fd = fileno(fp);
    const size_t buf_size = n_buffered_events[tid] *
        sizeof(hclib_instrument_event);

    // Wait for any pending AIO
    if (flushing_cb.aio_nbytes > 0) {
        int status;
        do {
            status = aio_error(&flushing_cb);
            assert(status == EINPROGRESS || status == 0);
        } while (status == EINPROGRESS);

        const int err = aio_return(&flushing_cb);
        assert(err != -1);
    }

    memset(&flushing_cb, 0x00, sizeof(flushing_cb));
    if (buf_size > 0) {
        flushing_cb.aio_fildes = fd;
        flushing_cb.aio_offset = dump_file_offsets[tid];
        flushing_cb.aio_buf = active_thread_buffers[tid];
        flushing_cb.aio_nbytes = buf_size;
        aio_write(&flushing_cb);

        dump_file_offsets[tid] += buf_size;
        hclib_instrument_event *tmp = active_thread_buffers[tid];
        active_thread_buffers[tid] = flushing_thread_buffers[tid];
        flushing_thread_buffers[tid] = tmp;
        n_buffered_events[tid] = 0;
    }
}

int register_event_type(char *event_name) {
    event_types = (hclib_event_type_info *)realloc(event_types,
            (n_event_types + 1) * sizeof(hclib_event_type_info));
    assert(event_types);

    event_types[n_event_types].event_type = n_event_types;

    event_types[n_event_types].name = (char *)malloc(strlen(event_name) + 1);
    assert(event_types[n_event_types].name);
    memcpy(event_types[n_event_types].name, event_name, strlen(event_name) + 1);

    n_event_types++;

    return n_event_types - 1;
}

static void write_dump_file_header(FILE *fp, const unsigned tid) {
    char buf[1024];
    int i;

    sprintf(buf, "%d\n", n_event_types);
    fprintf(fp, "%s", buf);
    dump_file_offsets[tid] += strlen(buf);

    for (i = 0; i < n_event_types; i++) {
        hclib_event_type_info *type = event_types + i;

        sprintf(buf, "%u %s\n", type->event_type, type->name);
        fprintf(fp, "%s", buf);
        dump_file_offsets[tid] += strlen(buf);
    }
}

/*
 * Assume this is called after all calls to register_event_type, which generally
 * requires that modules use register_event_type from their pre-initialization
 * callbacks.
 */
void initialize_instrumentation(const unsigned nthreads) {
    unsigned i;
    char filename[1024];

    active_thread_buffers = (hclib_instrument_event **)malloc(
            nthreads * sizeof(hclib_instrument_event *));
    flushing_thread_buffers = (hclib_instrument_event **)malloc(
            nthreads * sizeof(hclib_instrument_event *));
    assert(active_thread_buffers && flushing_thread_buffers);

    n_buffered_events = (unsigned *)malloc(nthreads * sizeof(unsigned));
    assert(n_buffered_events);

    memset(n_buffered_events, 0x00, nthreads * sizeof(unsigned));
    for (i = 0; i < nthreads; i++) {
        active_thread_buffers[i] = (hclib_instrument_event *)malloc(
                EVENT_BUFFER_LENGTH * sizeof(hclib_instrument_event));
        flushing_thread_buffers[i] = (hclib_instrument_event *)malloc(
                EVENT_BUFFER_LENGTH * sizeof(hclib_instrument_event));
        assert(active_thread_buffers[i] && flushing_thread_buffers[i]);
    }

    char *dump_file_dir = getenv("HCLIB_DUMP_DIR");
    if (dump_file_dir == NULL) {
        dump_file_dir = "/tmp";
    }
    assert(dump_file_dir[strlen(dump_file_dir) - 1] != '/');

    const unsigned long long current_time = hclib_current_time_ns();
    dump_folder = (char *)malloc(512);
    assert(dump_folder);
    sprintf(dump_folder, "%s/hclib.%llu.dump", dump_file_dir, current_time);
    const int mkdir_err = mkdir(dump_folder, 0700);
    assert(mkdir_err == 0);

    dump_files = (FILE **)malloc(nthreads * sizeof(FILE *));
    assert(dump_files);
    dump_file_offsets = (off_t *)malloc(nthreads * sizeof(off_t));
    assert(dump_file_offsets);
    memset(dump_file_offsets, 0x00, nthreads * sizeof(off_t));
    for (i = 0; i < nthreads; i++) {
        sprintf(filename, "%s/%d", dump_folder, i);
        dump_files[i] = fopen(filename, "w");
        if (dump_files[i] == NULL) {
            fprintf(stderr, "Failed creating dump file %s for thread %d : %s\n",
                    filename, i, strerror(errno));
            exit(1);
        }
        write_dump_file_header(dump_files[i], i);
    }

    thread_event_counters = (unsigned *)malloc(nthreads * sizeof(unsigned));
    assert(thread_event_counters);
    memset(thread_event_counters, 0x00, nthreads * sizeof(unsigned));

    save_nthreads = nthreads;
    flushing_cb.aio_nbytes = 0;
}

void finalize_instrumentation() {
    int i;

    for (i = 0; i < save_nthreads; i++) {
        // Two flushes in a row ensures all data gets out to disk
        flush_events(i);
        flush_events(i);

        fclose(dump_files[i]);

        free(active_thread_buffers[i]);
        free(flushing_thread_buffers[i]);
    }

    free(active_thread_buffers);
    free(flushing_thread_buffers);
    free(n_buffered_events);
    free(dump_files);
    free(thread_event_counters);

    for (i = 0; i < n_event_types; i++) {
        free(event_types[i].name);
    }
    free(event_types);

    fprintf(stderr, "HClib instrumentation dumped to %s\n", dump_folder);
    free(dump_folder);
}

unsigned hclib_register_event(const unsigned event_type,
        event_transition transition, const int event_id) {
    const unsigned long long timestamp = hclib_current_time_ns();
    const int tid = CURRENT_WS_INTERNAL->id;

    if (n_buffered_events[tid] == EVENT_BUFFER_LENGTH) {
        /*
         * Write out. This could be made ALOT more efficient. Keeping it simple
         * for now to get a prototype working.
         */
        flush_events(tid);
    }
    const int buf_index = n_buffered_events[tid];
    assert(buf_index < EVENT_BUFFER_LENGTH);
    hclib_instrument_event *event_buf = active_thread_buffers[tid];
    event_buf[buf_index].timestamp_ns = timestamp;
    event_buf[buf_index].event_type = event_type;
    event_buf[buf_index].transition = transition;

    int my_event_id;
    if (transition == START) {
        assert(event_id < 0);
        my_event_id = thread_event_counters[tid];
        thread_event_counters[tid] += 1;
    } else {
        assert(event_id >= 0);
        my_event_id = event_id;
    }
    event_buf[buf_index].event_id = my_event_id;

    n_buffered_events[tid] = buf_index + 1;

    return my_event_id;
}
