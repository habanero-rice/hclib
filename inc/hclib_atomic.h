#ifndef HCLIB_ATOMIC_H
#define HCLIB_ATOMIC_H

#include "hclib-rt.h"

#define CACHE_LINE_LEN_IN_BYTES 32

// C APIs

/*
 * User-defined callback for initializing a given atomic value. Accepts a
 * pointer to a block of memory used to store an atomic value and some optional
 * user data.
 */
typedef void (*atomic_init_func)(void *atomic_ele, void *user_data);

/*
 * User-defined callback for atomically updating a given atomic value. Accepts a
 * pointer to a block of memory used to store the atomic value and some optional
 * user data.
 */
typedef void (*atomic_update_func)(void *atomic_ele, void *user_data);

/*
 * User-defined callback for atomically reducing two values. Accepts two
 * pointers to blocks of memory used to store atomic values and some optional
 * user data. This function should reduce the values stored at a and b into the
 * memory pointed to by a.
 */
typedef void (*atomic_gather_func)(void *a, void *b, void *user_data);

/*
 * Storage for an atomic value. A partially updated copy of the final atomic
 * value produced is kept for each thread. The stored values are padded to avoid
 * false sharing.
 */
typedef struct _hclib_atomic_t {
    char *vals;
    size_t nthreads;
    size_t val_size;
    size_t padded_val_size;
} hclib_atomic_t;

/*
 * Create an atomic variable, including allocating memory for the user.
 */
extern hclib_atomic_t *hclib_atomic_create(const size_t ele_size_in_bytes,
        atomic_init_func init, void *user_data);

/*
 * Initialize pre-allocated storage with an atomic variable.
 */
extern void hclib_atomic_init(hclib_atomic_t *atomic,
        const size_t ele_size_in_bytes, atomic_init_func init, void *user_data);

/**
 * Atomically update the contents of an atomic variable using the user-defined
 * update function. The update function is passed user_data unchanged.
 */
extern void hclib_atomic_update(hclib_atomic_t *atomic, atomic_update_func f,
        void *user_data);

/**
 * Atomically gather/reduce partial atomic values to produce a final atomic
 * using the user-defined gather function. The gather function is passed
 * user_data unchanged.
 */
extern void *hclib_atomic_gather(hclib_atomic_t *atomic, atomic_gather_func f,
        void *user_data);

// C++ APIs
#ifdef __cplusplus

#include <functional>

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
            nthreads = hclib_get_num_workers();

            vals = (padded_val_t *)malloc(nthreads * sizeof(padded_val_t));
            assert(vals);
            for (unsigned i = 0; i < nthreads; i++) {
                vals[i].val = default_value;
            }
        }

        // Copy constructor
        atomic_t(const atomic_t &other) {
            nthreads = other.nthreads;
            vals = (padded_val_t *)malloc(nthreads * sizeof(padded_val_t));
            assert(vals);
            memcpy(vals, other.vals, nthreads * sizeof(padded_val_t));
            default_value = other.default_value;
        }

        // Destructor
        ~atomic_t() {
            free(vals);
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
#endif // __cplusplus

#endif
