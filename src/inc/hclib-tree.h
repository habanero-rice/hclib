#ifndef _HCLIB_TREE_H
#define _HCLIB_TREE_H

#include <stdlib.h>

#define LEFT 0
#define RIGHT 1

typedef struct _hclib_memory_tree_node {
    int height;
    struct _hclib_memory_tree_node *children[2];

    unsigned char *start_address;
    size_t length;
} hclib_memory_tree_node;

extern void hclib_memory_tree_insert(void *address, size_t length,
        hclib_memory_tree_node **root);
extern void hclib_memory_tree_remove(void *address,
        hclib_memory_tree_node **root);
extern int hclib_memory_tree_contains(void *address,
        hclib_memory_tree_node **root);

#endif
