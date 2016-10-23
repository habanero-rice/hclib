TARGET=bfs_oshmem bfs_oshmem-mailbox bfs_oshmem-single-mailbox bfs_oshmem-single-mailbox-omp bfs_oshmem-single-mailbox-replicated bfs_oshmem-single-mailbox-strided bfs_oshmem-single-mailbox-concurrent

all: $(TARGET)

bfs_oshmem: main.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3

bfs_oshmem-mailbox: main-mailbox.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox: main-single-mailbox.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-concurrent: main-single-mailbox-concurrent.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-strided: main-single-mailbox-strided.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-replicated: main-single-mailbox-replicated.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-omp: main-single-mailbox-omp.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -fopenmp -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

clean:
	rm -f $(TARGET)
