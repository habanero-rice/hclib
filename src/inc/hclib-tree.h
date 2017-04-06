/*
 * Copyright 2017 Rice University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
