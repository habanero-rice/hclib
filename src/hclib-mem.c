#include <string.h>

#include "hclib.h"
#include "hclib-rt.h"
#include "hclib-task.h"
#include "hclib-async-struct.h"
#include "hclib-finish.h"
#include "hclib-module.h"
#include "hclib-fptr-list.h"

#ifdef __cplusplus
extern "C" {
#endif

// A list of allocation functions registered for specific locale types
static hclib_fptr_list_t *alloc_registrations = NULL;
static hclib_fptr_list_t *realloc_registrations = NULL;
static hclib_fptr_list_t *free_registrations = NULL;
static hclib_fptr_list_t *memset_registrations = NULL;
static hclib_fptr_list_t *copy_registrations = NULL;

// Register a particular allocation function for a particular locale type
void hclib_register_alloc_func(int locale_id,
        hclib_module_alloc_impl_func_type func) {
    hclib_register_func(&alloc_registrations, locale_id, func, MAY_USE);
}

// Register a particular reallocation function for a particular locale type
void hclib_register_realloc_func(int locale_id,
        hclib_module_realloc_impl_func_type func) {
    hclib_register_func(&realloc_registrations, locale_id, func, MAY_USE);
}

// Register a particular free-ing function for a particular locale type
void hclib_register_free_func(int locale_id,
        hclib_module_free_impl_func_type func) {
    hclib_register_func(&free_registrations, locale_id, func, MAY_USE);
}

// Register a particular memset function for a particular locale type
void hclib_register_memset_func(int locale_id,
        hclib_module_memset_impl_func_type func) {
    hclib_register_func(&memset_registrations, locale_id, func, MAY_USE);
}

// Register a particular copy function for a particular locale type
void hclib_register_copy_func(int locale_id,
        hclib_module_copy_impl_func_type func, int priority) {
    hclib_register_func(&copy_registrations, locale_id, func, priority);
}

typedef struct _malloc_struct {
    size_t nbytes;
    hclib_locale_t *locale;
    hclib_promise_t *promise;
    hclib_module_alloc_impl_func_type cb;
} malloc_struct;

static void allocate_kernel(void *arg) {
    malloc_struct *ms = (malloc_struct *)arg;
    void *allocated = (ms->cb)(ms->nbytes, ms->locale);
    hclib_promise_put(ms->promise, allocated);
    free(ms);
}

hclib_future_t *hclib_allocate_at(size_t nbytes, hclib_locale_t *locale) {
    assert(hclib_has_func_for(alloc_registrations, locale->type));

    hclib_promise_t *promise = hclib_promise_create();

    malloc_struct *ms = (malloc_struct *)malloc(sizeof(malloc_struct));
    ms->nbytes = nbytes;
    ms->locale = locale;
    ms->promise = promise;
    ms->cb = hclib_get_func_for(alloc_registrations, locale->type);

    hclib_async(allocate_kernel, ms, NULL, NULL, locale);
    return hclib_get_future_for_promise(promise);
}

typedef struct _realloc_struct {
    void *ptr;
    size_t nbytes;
    hclib_locale_t *locale;
    hclib_promise_t *promise;
    hclib_module_realloc_impl_func_type cb;
} realloc_struct;

static void reallocate_kernel(void *arg) {
    realloc_struct *rs = (realloc_struct *)arg;
    void *reallocated = (rs->cb)(rs->ptr, rs->nbytes, rs->locale);
    hclib_promise_put(rs->promise, reallocated);
    free(rs);
}

hclib_future_t *hclib_reallocate_at(void *ptr, size_t new_nbytes,
        hclib_locale_t *locale) {
    assert(hclib_has_func_for(realloc_registrations, locale->type));

    hclib_promise_t *promise = hclib_promise_create();

    realloc_struct *rs = (realloc_struct *)malloc(sizeof(realloc_struct));
    rs->ptr = ptr;
    rs->nbytes = new_nbytes;
    rs->locale = locale;
    rs->promise = promise;
    rs->cb = hclib_get_func_for(realloc_registrations, locale->type);

    hclib_async(reallocate_kernel, rs, NULL, NULL, locale);
    return hclib_get_future_for_promise(promise);
}

typedef struct _memset_struct {
    void *ptr;
    size_t nbytes;
    int pattern;
    hclib_locale_t *locale;
    hclib_promise_t *promise;
    hclib_module_memset_impl_func_type cb;
} memset_struct;

static void memset_kernel(void *arg) {
    memset_struct *ms = (memset_struct *)arg;
    (ms->cb)(ms->ptr, ms->pattern, ms->nbytes, ms->locale);
    hclib_promise_put(ms->promise, NULL);
    free(ms);
}

hclib_future_t *hclib_memset_at(void *ptr, int pattern, size_t nbytes,
        hclib_locale_t *locale) {
    assert(hclib_has_func_for(memset_registrations, locale->type));

    hclib_promise_t *promise = hclib_promise_create();

    memset_struct *ms = (memset_struct *)malloc(sizeof(memset_struct));
    ms->ptr = ptr;
    ms->nbytes = nbytes;
    ms->pattern = pattern;
    ms->locale = locale;
    ms->promise = promise;
    ms->cb = hclib_get_func_for(memset_registrations, locale->type);

    hclib_async(memset_kernel, ms, NULL, NULL, locale);
    return hclib_get_future_for_promise(promise);
}

// TODO at some point it may be useful to have this as a future too
void hclib_free_at(void *ptr, hclib_locale_t *locale) {
    assert(hclib_has_func_for(free_registrations, locale->type));
    hclib_module_free_impl_func_type func =
        (hclib_module_free_impl_func_type)hclib_get_func_for(free_registrations,
                locale->type);
    func(ptr, locale);
}

typedef struct _copy_struct {
    hclib_locale_t *dst_locale;
    void *dst;
    hclib_locale_t *src_locale;
    void *src;
    size_t nbytes;
    hclib_promise_t *promise;
    hclib_module_copy_impl_func_type cb;
} copy_struct;

static void copy_kernel(void *arg) {
    copy_struct *cs = (copy_struct *)arg;
    (cs->cb)(cs->dst_locale, cs->dst, cs->src_locale, cs->src, cs->nbytes);
    hclib_promise_put(cs->promise, NULL);
    free(cs);
}

hclib_future_t *hclib_async_copy(hclib_locale_t *dst_locale, void *dst,
        hclib_locale_t *src_locale, void *src, size_t nbytes) {
    hclib_promise_t *promise = hclib_promise_create();

    hclib_module_copy_impl_func_type dst_cb = hclib_get_func_for(
            copy_registrations, dst_locale->type);
    hclib_module_copy_impl_func_type src_cb = hclib_get_func_for(
            copy_registrations, src_locale->type);
    int dst_priority = hclib_get_priority_for(copy_registrations,
            dst_locale->type);
    int src_priority = hclib_get_priority_for(copy_registrations,
            src_locale->type);
    hclib_module_copy_impl_func_type copy_cb;
    assert(dst_cb != NULL || src_cb != NULL);
    if (dst_cb == NULL) {
        copy_cb = src_cb;
    } else if (src_cb == NULL) {
        copy_cb = dst_cb;
    } else {
        // Not both MUST_USE
        assert(!(dst_priority == MUST_USE && src_priority == MUST_USE));
        if (src_priority == MUST_USE) copy_cb = src_cb;
        else copy_cb = dst_cb;
    }

    copy_struct *cs = (copy_struct *)malloc(sizeof(copy_struct));
    cs->dst_locale = dst_locale;
    cs->dst = dst;
    cs->src_locale = src_locale;
    cs->src = src;
    cs->nbytes = nbytes;
    cs->promise = promise;
    cs->cb = copy_cb;

    hclib_async(copy_kernel, cs, NULL, NULL, dst_locale);
    return hclib_get_future_for_promise(promise);
}
