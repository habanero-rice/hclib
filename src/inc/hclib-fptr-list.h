#ifndef HCLIB_FPTR_LIST_H
#define HCLIB_FPTR_LIST_H

typedef struct _hclib_fptr_list_t {
    void **fptrs;
    int *priorities; // MUST_USE or MAY_USE
    int capacity;
} hclib_fptr_list_t;

void hclib_register_func(hclib_fptr_list_t **list, int index, void *fptr,
        int priority);
int hclib_has_func_for(hclib_fptr_list_t *list, int index);
void *hclib_get_func_for(hclib_fptr_list_t *list, int index);
int hclib_get_priority_for(hclib_fptr_list_t *list, int index);

#endif
