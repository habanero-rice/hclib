#ifndef HCLIB_MODULE_H
#define HCLIB_MODULE_H

#include "hclib-locality-graph.h"

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

#define HCLIB_MODULE_INITIALIZATION_FUNC(module_init_funcname) void module_init_funcname()
#define HCLIB_REGISTER_MODULE(module_name,module_pre_init_func,module_post_init_func) const static int ____hclib_module_init = hclib_add_module_init_function(module_name, module_pre_init_func, module_post_init_func);

#ifdef __cplusplus
namespace hclib {
#endif

    // TODO user-faceing C++ APIs?

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
int hclib_add_module_init_function(const char *lbl,
        hclib_module_pre_init_func_type pre,
        hclib_module_post_init_func_type post);

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

void hclib_call_module_pre_init_functions();
void hclib_call_module_post_init_functions();
#ifdef __cplusplus
}
#endif

#endif
