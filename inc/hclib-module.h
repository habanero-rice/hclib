#ifndef HCLIB_MODULE_H
#define HCLIB_MODULE_H

#include "hclib-locality-graph.h"

/*
 * In some cases, multiple modules may have candidate functions for a particular
 * operation (e.g. when doing a copy between an address space owned by one
 * module to another). In those cases, we use MUST_USE to identify modules whose
 * callbacks must be used for a particular operation involving it and MAY_USE to
 * identify modules that are to be used only as a fallback if there are no
 * MUST_USE modules.
 */
#define MUST_USE 1
#define MAY_USE 2

/*
 * Type for functions called for each module before the HClib runtime is
 * initialized.
 */
typedef void (*hclib_module_pre_init_func_type)();

/*
 * Type for functions called for each module after the HClib runtime is
 * initialized.
 */
typedef void (*hclib_module_post_init_func_type)();

/*
 * Type for finalize functions for each module.
 */
typedef void (*hclib_module_finalize_func_type)();

/*
 * Callbacks on locale creation to allow for insertion of module-specific
 * metadata.
 */
typedef size_t (*hclib_locale_metadata_size_func_type)();
typedef void (*hclib_locale_metadata_populate_func_type)(hclib_locale_t *);

// Type for functions to perform allocations at a given locale.
typedef void * (*hclib_module_alloc_impl_func_type)(size_t, hclib_locale_t *);
typedef void * (*hclib_module_realloc_impl_func_type)(void *, size_t,
        hclib_locale_t *);
typedef void (*hclib_module_free_impl_func_type)(void *, hclib_locale_t *);
typedef void (*hclib_module_memset_impl_func_type)(void *, int, size_t,
        hclib_locale_t *);
typedef void (*hclib_module_copy_impl_func_type)(hclib_locale_t *, void *,
        hclib_locale_t *, void *, size_t);

#define HCLIB_MODULE_INITIALIZATION_FUNC(module_init_funcname) void module_init_funcname()
#define HCLIB_REGISTER_MODULE(module_name,module_pre_init_func,module_post_init_func,module_finalize_func) const static int ____hclib_module_init = hclib_add_module_init_function(module_name, module_pre_init_func, module_post_init_func, module_finalize_func);

#ifdef __cplusplus
namespace hclib {
#endif

    // TODO user-facing C++ APIs?

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
int hclib_add_module_init_function(const char *lbl,
        hclib_module_pre_init_func_type pre,
        hclib_module_post_init_func_type post,
        hclib_module_finalize_func_type finalize);

void hclib_add_locale_metadata_functions(int locale_id,
        hclib_locale_metadata_size_func_type size_func,
        hclib_locale_metadata_populate_func_type populate_func);

void hclib_register_alloc_func(int locale_id,
        hclib_module_alloc_impl_func_type func);
void hclib_register_realloc_func(int locale_id,
        hclib_module_realloc_impl_func_type func);
void hclib_register_free_func(int locale_id,
        hclib_module_free_impl_func_type func);
void hclib_register_memset_func(int locale_id,
        hclib_module_memset_impl_func_type func);
void hclib_register_copy_func(int locale_id,
        hclib_module_copy_impl_func_type func, int priority);

void hclib_call_module_pre_init_functions();
void hclib_call_module_post_init_functions();
void hclib_call_finalize_functions();
#ifdef __cplusplus
}
#endif

#endif
