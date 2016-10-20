TARGET=bfs_oshmem bfs_oshmem-mailbox bfs_oshmem-single-mailbox

all: $(TARGET)

bfs_oshmem: main.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3

bfs_oshmem-mailbox: main-mailbox.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox: main-single-mailbox.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

clean:
	rm -f $(TARGET)
