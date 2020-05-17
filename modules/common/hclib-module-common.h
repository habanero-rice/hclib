#ifndef HCLIB_MODULE_COMMON_H
#define HCLIB_MODULE_COMMON_H

#ifdef HCLIB_INSTRUMENT
#include "hclib-instrument.h"
#endif

namespace hclib {

template<class pending_op>
void poll_on_pending(pending_op **addr_of_head,
        bool (*test_completion_callback)(void *),
        hclib::locale_t *locale_to_yield_to) {
    do {
        int pending_list_non_empty = 1;

        pending_op *prev = NULL;
        pending_op *op = *addr_of_head;

        assert(op != NULL);

        while (op) {
            pending_op *next = op->next;

            const bool complete = test_completion_callback(op);

            if (complete) {
                // Remove from singly linked list
                if (prev == NULL) {
                    /*
                     * If previous is NULL, we *may* be looking at the front of
                     * the list. It is also possible that another thread in the
                     * meantime came along and added an entry to the front of
                     * this singly-linked wait list, in which case we need to
                     * ensure we update its next rather than updating the list
                     * head. We do this by first trying to automatically update
                     * the list head to be the next of wait_set, and if we fail
                     * then we know we have a new head whose next points to
                     * wait_set and which should be updated.
                     */
                    pending_op *old_head = __sync_val_compare_and_swap(
                            addr_of_head, op, next);
                    if (old_head != op) {
                        // Failed, someone else added a different head
                        assert(old_head->next == op);
                        old_head->next = next;
                        prev = old_head;
                    } else {
                        /*
                         * Success, new head is now wait_set->next. We want this
                         * polling task to exit if we just set the head to NULL.
                         * It is the responsibility of future async_when calls
                         * to restart it upon discovering a null head.
                         */
                        pending_list_non_empty = (next != NULL);
                    }
                } else {
                    /*
                     * If previous is non-null, we just adjust its next link to
                     * jump over the current node.
                     */
                    assert(prev->next == op);
                    prev->next = next;
                }

#ifdef HCLIB_INSTRUMENT
                hclib_register_event(op->event_type, END, op->event_id);
#endif

                if (op->prom) {
                    op->prom->put();
                } else {
                    spawn(op->task);
                }
                free(op);
            } else {
                prev = op;
            }

            op = next;
        }

        if (pending_list_non_empty) {
            hclib::yield_at(locale_to_yield_to);
        } else {
            // Empty list
            break;
        }
    } while (true);
}

template<class pending_op>
void append_to_pending(pending_op *op, pending_op **addr_of_head,
        bool (*test_completion_callback)(void *),
        hclib::locale_t *locale_to_yield_to) {
    pending_op *pending = *addr_of_head;
    op->next = pending;

    pending_op *old_head;
    while (1) {
        old_head = __sync_val_compare_and_swap(addr_of_head, op->next, op);
        if (old_head != op->next) {
            op->next = old_head;
        } else {
            break;
        }
    }

    if (old_head == NULL) {
        hclib::async_at([addr_of_head, test_completion_callback, locale_to_yield_to] {
            hclib::poll_on_pending<pending_op>(addr_of_head,
                test_completion_callback, locale_to_yield_to);
        }, locale_to_yield_to);
    }
}

}

#endif
