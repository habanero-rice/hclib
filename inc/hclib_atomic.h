#ifndef HCLIB_ATOMIC_H
#define HCLIB_ATOMIC_H

#include <functional>

#include "hclib-rt.h"

#define CACHE_LINE_LEN_IN_BYTES 32

/**
 * A base atomic variable for use in HClib programs.
 */
template <class T>
class hclib_atomic_t {
    private:
        typedef struct _padded_val_t {
            T val;
            char padding[CACHE_LINE_LEN_IN_BYTES - sizeof(T)];
        } padded_val_t;

        size_t nthreads;
        padded_val_t *vals;
        T default_value;

    public:
        hclib_atomic_t(T set_default_value) {
            default_value = set_default_value;
            nthreads = hclib_num_workers();

            vals = (padded_val_t *)malloc(nthreads * sizeof(padded_val_t));
            for (int i = 0; i < nthreads; i++) {
                vals[i].val = default_value;
            }
        }

        void update(std::function<T(T)> f) {
            const int wid = hclib_get_current_worker();
            vals[wid].val = f(vals[wid].val);
        }

        /**
         * Gather the results of all threads together and return them using a
         * programmer-provided reduction function. Note that it is the
         * programmer's responsibility to ensure that this atomic variable is no
         * longer being updated at the time gather is called.
         */
        T gather(std::function<T(T, T)> reduce) {
            T aggregate = default_value;
            for (int i = 0; i < nthreads; i++) {
                aggregate = reduce(aggregate, vals[i].val);
            }
            return aggregate;
        }
};

#endif
