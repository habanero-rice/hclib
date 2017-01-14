#ifndef HCLIB_INSTRUMENT_H
#define HCLIB_INSTRUMENT_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A class representing a single event in the HClib runtime.
 */
typedef struct _hclib_instrument_event {
    // The time at which the event occurred
    unsigned long long timestamp_ns;

    // The type of event, e.g. MPI_SEND_START, MPI_ISEND_START
    unsigned event_type;

    // A thread-local unique ID for this event
    unsigned event_id;
} hclib_instrument_event;

/*
 * Metadata on a specific event type.
 */
typedef struct _hclib_event_type_info {
    /*
     * The unique ID that was given to this event, correlates with the
     * event_type field in hclib_instrument_event.
     */
    unsigned event_type;

    // Human-readable name for this event
    char *name;
} hclib_event_type_info;

typedef enum _event_transition {
    START,
    END
} event_transition;

/*
 * Register a single event type (e.g. MPI_SEND_START) with the HClib runtime,
 * returning a unique integer ID for this event type.
 *
 * Note: register_event_type is not thread-safe.
 */
int register_event_type(char *event_name);

/*
 * Set up data structures for instrumentation.
 */
void initialize_instrumentation(const unsigned nthreads);

void finalize_instrumentation();

unsigned hclib_register_event(const unsigned event_type,
        event_transition transition, const int event_id);

#ifdef __cplusplus
}
#endif

#endif // HCLIB_INSTRUMENT_H
