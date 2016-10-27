#ifndef HCLIB_ATOMIC_H
#define HCLIB_ATOMIC_H

#include <functional>

#include "hclib-rt.h"

#define CACHE_LINE_LEN_IN_BYTES 32

namespace hclib {

/**
 * A base atomic variable for use in HClib programs.
 */
template <class T>
class atomic_t {
    private:
        typedef struct _padded_val_t {
            T val;
            char padding[CACHE_LINE_LEN_IN_BYTES - sizeof(T)];
        } padded_val_t;

        size_t nthreads;
        padded_val_t *vals;
        T default_value;

    public:
        atomic_t(T set_default_value) {
            default_value = set_default_value;
            nthreads = hclib_num_workers();

            vals = (padded_val_t *)malloc(nthreads * sizeof(padded_val_t));
            for (unsigned i = 0; i < nthreads; i++) {
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
            for (unsigned i = 0; i < nthreads; i++) {
                aggregate = reduce(aggregate, vals[i].val);
            }
            return aggregate;
        }
};

// Provide some commonly useful atomic variables below

template <class T>
class atomic_sum_t : private atomic_t<T> {
    public:
        atomic_sum_t(T set_default_value) : atomic_t<T>(set_default_value) {
        }

        atomic_sum_t& operator+=(T other) {
            atomic_t<T>::update([&other] (T curr) { return curr + other; });
            return *this;
        }

        T get() {
            return atomic_t<T>::gather([] (T a, T b) { return a + b; });
        }
};

template <class T>
class atomic_max_t : private atomic_t<T> {
    public:
        atomic_max_t(T set_default_value) : atomic_t<T>(set_default_value) {
        }

        void update(T other) {
            atomic_t<T>::update([&other] (T curr) { return (curr > other ? curr : other); });
        }

        T get() {
            return atomic_t<T>::gather([] (T a, T b) { return (a > b ? a : b); });
        }
};

template <class T>
class atomic_or_t : private atomic_t<T> {
    public:
        atomic_or_t(T set_default_value) : atomic_t<T>(set_default_value) {
        }

        atomic_or_t& operator|=(T other) {
            atomic_t<T>::update([&other] (T curr) { return curr || other; });
            return *this;
        }

        T get() {
            return atomic_t<T>::gather([] (T a, T b) { return a || b; });
        }
};


}

#endif
