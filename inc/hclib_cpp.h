#ifndef HCLIB_CPP_H_
#define HCLIB_CPP_H_

#include "hclib_common.h"
#include "hclib-rt.h"
#include "hclib-async.h"
#include "hclib-forasync.h"
#include "hclib_promise.h"
#include "hclib.h"
#include "hclib-locality-graph.h"

namespace hclib {

typedef hclib_locale_t locale_t;

template <typename T>
inline void launch(const char **deps, int ndeps, T &&lambda) {
    typedef typename std::remove_reference<T>::type U;
    hclib_task_t *user_task = _allocate_async(new U(lambda));
    hclib_launch((generic_frame_ptr)spawn_root, user_task, deps, ndeps);
}

template <typename T>
inline void launch(const int nworkers, const char **deps, int ndeps,
        T &&lambda) {
    typedef typename std::remove_reference<T>::type U;
    hclib_task_t *user_task = _allocate_async(new U(lambda));

    char nworkers_str[32];
    sprintf(nworkers_str, "%d", nworkers);
    setenv("HCLIB_WORKERS", nworkers_str, 1);

    hclib_launch((generic_frame_ptr)spawn_root, user_task, deps, ndeps);
}

extern hclib_worker_state *current_ws();
int get_current_worker();
int get_num_workers();

int get_num_locales();
hclib_locale_t *get_closest_locale();
hclib_locale_t **get_thread_private_locales();
hclib_locale_t *get_all_locales();
hclib_locale_t **get_all_locales_of_type(int type, int *out_count);
hclib_locale_t *get_master_place();

hclib::future_t<void*> *allocate_at(size_t nbytes, hclib::locale_t *locale);
hclib::future_t<void*> *reallocate_at(void *ptr, size_t nbytes,
        hclib::locale_t *locale);
void free_at(void *ptr, hclib::locale_t *locale);
hclib::future_t<void*> *memset_at(void *ptr, int pattern, size_t nbytes,
        hclib::locale_t *locale);

inline hclib::future_t<void*> *async_copy(hclib::locale_t *dst_locale,
        void *dst, hclib::locale_t *src_locale, void *src, size_t nbytes) {
    hclib_future_t *actual = hclib_async_copy(dst_locale, dst, src_locale, src,
            nbytes, NULL, 0);
    return (hclib::future_t<void *> *)actual;
}

inline hclib::future_t<void*> *async_copy_await(hclib::locale_t *dst_locale,
        void *dst, hclib::locale_t *src_locale, void *src, size_t nbytes,
        hclib_future_t *future) {
    hclib_future_t *actual = hclib_async_copy(dst_locale, dst, src_locale, src,
            nbytes, future ? &future : NULL, future ? 1 : 0);
    return (hclib::future_t<void *> *)actual;
}

inline hclib::future_t<void*> *async_copy_await_all(hclib::locale_t *dst_locale,
        void *dst, hclib::locale_t *src_locale, void *src, size_t nbytes,
        hclib_future_t **futures, const int nfutures) {
    hclib_future_t *actual = hclib_async_copy(dst_locale, dst, src_locale, src,
            nbytes, futures, nfutures);
    return (hclib::future_t<void *> *)actual;
}

}

#endif
