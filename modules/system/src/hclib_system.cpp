#include "hclib_system.h"
#include "hclib-locality-graph.h"

#include <iostream>

static int l1_locale_id, l2_locale_id, l3_locale_id, sysmem_locale_id;

static void *allocation_func(size_t nbytes, hclib_locale_t *locale) {
    void *ptr = malloc(nbytes);
#ifdef VERBOSE
    std::cerr << __func__ << " ptr=" << ptr << " nbytes=" << nbytes <<
        " locale=" << locale << std::endl;
#endif
    if (ptr) memset(ptr, 42, 1); // first touch allocation
    return ptr;
}

static void *reallocation_func(void *old, size_t nbytes,
        hclib_locale_t *locale) {
#ifdef VERBOSE
    std::cerr << __func__ << " old=" << old << " nbytes=" << nbytes <<
        " locale=" << locale << std::endl;
#endif
    void *ptr = realloc(old, nbytes);
    return ptr;
}

static void free_func(void *ptr, hclib_locale_t *locale) {
#ifdef VERBOSE
    std::cerr << __func__ << " ptr=" << ptr << " locale=" << locale <<
        std::endl;
#endif
    free(ptr);
}

static void memset_func(void *ptr, int val, size_t nbytes,
        hclib_locale_t *locale) {
#ifdef VERBOSE
    std::cerr << __func__ << " ptr=" << ptr << " val=" << val << " nbytes=" <<
        nbytes << " locale=" << locale << std::endl;
#endif
    memset(ptr, val, nbytes);
}

static void copy_func(hclib_locale_t *dst_locale, void *dst,
        hclib_locale_t *src_locale, void *src, size_t nbytes) {
    memcpy(dst, src, nbytes);
}

HCLIB_MODULE_INITIALIZATION_FUNC(system_pre_initialize) {
    l1_locale_id = hclib_add_known_locale_type("L1");
    l2_locale_id = hclib_add_known_locale_type("L2");
    l3_locale_id = hclib_add_known_locale_type("L3");
    sysmem_locale_id = hclib_add_known_locale_type("sysmem");
}

HCLIB_MODULE_INITIALIZATION_FUNC(system_post_initialize) {
    hclib_register_alloc_func(l1_locale_id, allocation_func);
    hclib_register_alloc_func(l2_locale_id, allocation_func);
    hclib_register_alloc_func(l3_locale_id, allocation_func);
    hclib_register_alloc_func(sysmem_locale_id, allocation_func);

    hclib_register_realloc_func(l1_locale_id, reallocation_func);
    hclib_register_realloc_func(l2_locale_id, reallocation_func);
    hclib_register_realloc_func(l3_locale_id, reallocation_func);
    hclib_register_realloc_func(sysmem_locale_id, reallocation_func);

    hclib_register_free_func(l1_locale_id, free_func);
    hclib_register_free_func(l2_locale_id, free_func);
    hclib_register_free_func(l3_locale_id, free_func);
    hclib_register_free_func(sysmem_locale_id, free_func);

    hclib_register_memset_func(l1_locale_id, memset_func);
    hclib_register_memset_func(l2_locale_id, memset_func);
    hclib_register_memset_func(l3_locale_id, memset_func);
    hclib_register_memset_func(sysmem_locale_id, memset_func);

    hclib_register_copy_func(l1_locale_id, copy_func, MAY_USE);
    hclib_register_copy_func(l2_locale_id, copy_func, MAY_USE);
    hclib_register_copy_func(l3_locale_id, copy_func, MAY_USE);
    hclib_register_copy_func(sysmem_locale_id, copy_func, MAY_USE);
}

HCLIB_MODULE_INITIALIZATION_FUNC(system_finalize) {
}

hclib::locale_t *hclib::get_closest_cpu_locale() {
    int type_arr[4] = { l1_locale_id, l2_locale_id, l3_locale_id,
        sysmem_locale_id };
    return hclib_get_closest_locale_of_types(hclib_get_closest_locale(),
            type_arr, 4);
}

HCLIB_REGISTER_MODULE("system", system_pre_initialize, system_post_initialize, system_finalize)
