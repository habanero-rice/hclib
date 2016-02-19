#include "hclib-tree.h"
#include "hclib-rt.h"

#include <stdio.h>

// #define VERBOSE

/*
 * This self-balancing tree implementation is used to efficiently track pinned
 * memory ranges allocated for HClib GPU programs.
 */

static hclib_memory_tree_node *create_memory_tree_node(void *address,
        size_t length) {
    hclib_memory_tree_node *node = (hclib_memory_tree_node *)malloc(
            sizeof(hclib_memory_tree_node));
    HASSERT(node);

    node->start_address = address;
    node->length = length;
    node->height = -1;
    node->children[LEFT] = NULL;
    node->children[RIGHT] = NULL;

    return node;
}

static inline int height(hclib_memory_tree_node *t) {
    return t == NULL ? -1 : t->height;
}

static inline hclib_memory_tree_node *left(hclib_memory_tree_node *t) {
    return t->children[LEFT];
}

static inline hclib_memory_tree_node *right(hclib_memory_tree_node *t) {
    return t->children[RIGHT];
}

static inline int max(int a, int b) {
    return a > b ? a : b;
}

static void set_height(hclib_memory_tree_node *n) {
    n->height = 1 + max(height(left(n)), height(right(n)));
}

static int balance(hclib_memory_tree_node *n) {
    return height(left(n)) - height(right(n));
}

hclib_memory_tree_node *rotate(hclib_memory_tree_node **root, int dir) {
    hclib_memory_tree_node *old_r = *root;
    hclib_memory_tree_node *new_r = old_r->children[dir];

    if (NULL == (*root = new_r)) {
        free(old_r);
    } else {
        old_r->children[dir] = new_r->children[!dir];
        set_height(old_r);
        new_r->children[!dir] = old_r;
    }
    return new_r;
}
 
void adjust_balance(hclib_memory_tree_node **rootp) {
    hclib_memory_tree_node *root = *rootp;
    int b = balance(root) / 2;
    if (b) {
        int dir = (1 - b) / 2;
        if (balance(root->children[dir]) == -b) {
            rotate(&root->children[dir], !dir);
        }
        root = rotate(rootp, dir);
    }
    if (root != NULL) set_height(root);
}
 
// find the node that contains value as payload; or returns 0
static hclib_memory_tree_node *find(void *address,
        hclib_memory_tree_node *curr) {
    unsigned char *c_address = (unsigned char *)address;

    if (curr == NULL) {
        return NULL;
    }

    if (c_address >= curr->start_address &&
            c_address < curr->start_address + curr->length) {
        return curr;
    } else if (c_address < curr->start_address) {
        return find(address, left(curr));
    } else { // c_address >= curr->start_address + curr->length
        return find(address, right(curr));
    }
}
 
void hclib_memory_tree_insert(void *address, size_t length,
        hclib_memory_tree_node **rootp) {
    unsigned char *c_address = (unsigned char *)address;
    hclib_memory_tree_node *root = *rootp;

#ifdef VERBOSE
    fprintf(stderr, "hclib_memory_tree_insert: root=%p address=%p length=%lu\n",
            root, address, (unsigned long)length);
#endif

    if (root == NULL) {
        *rootp = create_memory_tree_node(address, length);
    } else {
        HASSERT(c_address < root->start_address ||
                c_address >= root->start_address + root->length);
        if (c_address < root->start_address) {
            hclib_memory_tree_insert(address, length, &root->children[LEFT]);
        } else {
            hclib_memory_tree_insert(address, length, &root->children[RIGHT]);
        }
        adjust_balance(rootp);
    }
}

void hclib_memory_tree_remove(void *address,
        hclib_memory_tree_node **rootp) {
    hclib_memory_tree_node *root = *rootp;
    unsigned char *c_address = (unsigned char *)address;
    HASSERT(root != NULL);

#ifdef VERBOSE
    fprintf(stderr, "hclib_memory_tree_remove: root=%p address=%p\n", root,
            address);
#endif

    // if this is the node we want, rotate until off the tree
    if (root->start_address == address) {
        if (NULL == (root = rotate(rootp, balance(root) < 0))) {
            return;
        }
    }

    hclib_memory_tree_remove(address,
            &root->children[c_address > root->start_address]);
    adjust_balance(rootp);
}

int hclib_memory_tree_contains(void *address, hclib_memory_tree_node **root) {
    if (*root == NULL) return 0;
    return find(address, *root) != NULL;
}
