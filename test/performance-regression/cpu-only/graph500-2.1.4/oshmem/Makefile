TARGET=bfs_oshmem bfs_oshmem-mailbox

all: $(TARGET)

bfs_oshmem: main.c utilities.c mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3

bfs_oshmem-mailbox: main-mailbox.c utilities.c mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3

clean:
	rm -f $(TARGET)
