#include "hclib-fptr-list.h"
#include "hclib-rt.h"
#include "hclib-module.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

void hclib_register_func(hclib_fptr_list_t **list, int index, void *fptr,
        int priority) {
    HASSERT(priority == MUST_USE || priority == MAY_USE);

    if (*list == NULL) {
        *list = (hclib_fptr_list_t *)malloc(sizeof(hclib_fptr_list_t));
        assert(*list);

        (*list)->fptrs = NULL;
        (*list)->priorities = NULL;
        (*list)->capacity = 0;
    }

    const size_t needed_capacity = index + 1;
    if (needed_capacity > (*list)->capacity) {
        (*list)->fptrs = (void **)realloc((*list)->fptrs,
                needed_capacity * sizeof(void *));
        (*list)->priorities = (int *)realloc((*list)->priorities,
                needed_capacity * sizeof(int));

        memset((*list)->fptrs + (*list)->capacity, 0x00,
                (needed_capacity - (*list)->capacity) * sizeof(void *));
        memset((*list)->priorities + (*list)->capacity, 0x00,
                (needed_capacity - (*list)->capacity) * sizeof(int));
        (*list)->capacity = needed_capacity;
    }

    HASSERT(((*list)->fptrs)[index] == NULL);
    ((*list)->fptrs)[index] = fptr;
    ((*list)->priorities)[index] = priority;
}

void *hclib_get_func_for(hclib_fptr_list_t *list, int index) {
    assert(list);
    if (index < list->capacity) {
        return (list->fptrs)[index];
    } else {
        return NULL;
    }
}

int hclib_has_func_for(hclib_fptr_list_t *list, int index) {
    return list != NULL && list->capacity > index && (list->fptrs)[index] != NULL;
}

int hclib_get_priority_for(hclib_fptr_list_t *list, int index) {
    assert(list);
    return (list->priorities)[index];
}
