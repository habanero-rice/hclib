/** Check the return of malloc function.
*/
#include "kmi.h"

static size_t memory_allocated = 0;

void *kmalloc(const char *file, int line, size_t size)
{
#if 0
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif
    memory_allocated += size;
    void *v = malloc(size);
    if (!v) {
#if 0
        if (myid == 0)
#endif
            fprintf(stderr, "Fail to allocate %lu :(%s:%d), total: %lu\n", size,
                    file, line, memory_allocated);
        exit(EXIT_FAILURE);
    } else {
        if (size > 1024)
#if 0
            if (myid == 0)
#endif
                printf("Allocate %lu :(%s:%d), total: %lu\n", size, file, line,
                       memory_allocated);
    }
    return v;
}

void kfree(const char *file, int line, void *ptr)
{
#if 0
    size_t *p = (size_t *) ptr;
    memory_allocated -= p[-1];
    if (p[-1] > 1024)
        printf("free %lu :(%s:%d), total:%lu\n", p[-1], file, line,
               memory_allocated);
#endif
    free(ptr);
}

void *krealloc(const char *file, int line, void *ptr, size_t size)
{
    void *v = realloc(ptr, size);
    if (!v) {
        fprintf(stderr, "Fail to reallocate %lu :(%s:%d)\n", size, file, line);
        exit(EXIT_FAILURE);
    }
    return v;
}
