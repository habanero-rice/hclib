TARGET=bfs_oshmem

$(TARGET): main.c utilities.c mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include -L$(OPENSHMEM_INSTALL)/lib64 -lsma -lrt -o $@ $^ -O3

clean:
	rm -f $(TARGET)
