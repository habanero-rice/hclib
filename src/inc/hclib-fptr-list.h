#ifndef HCLIB_FPTR_LIST_H
#define HCLIB_FPTR_LIST_H

typedef struct _hclib_fptr_list_t {
    void **fptrs;
    int capacity;
} hclib_fptr_list_t;

void hclib_register_func(hclib_fptr_list_t **list, int index, void *fptr);
void *hclib_get_func_for(hclib_fptr_list_t *list, int index);
int hclib_has_func_for(hclib_fptr_list_t *list, int index);

#endif
