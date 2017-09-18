TARGET=bfs_oshmem bfs_oshmem-mailbox bfs_oshmem-single-mailbox bfs_oshmem-single-mailbox-omp bfs_oshmem-single-mailbox-replicated bfs_oshmem-single-mailbox-strided bfs_oshmem-single-mailbox-concurrent bfs_oshmem-single-mailbox-hiper bfs_oshmem-single-mailbox-concurrent-crc bfs_oshmem-bitvector bfs_oshmem-single-mailbox-concurrent-crc-hiper

FLAGS=-ffast-math -O3 -g

# For Cray SHMEM
# OSHMEM_LD_FLAGS=-L$(OPENSHMEM_INSTALL)/lib64 -lsma

# For OpenMPI SHMEM
OSHMEM_LD_FLAGS=-L$(OPENSHMEM_INSTALL)/lib64 -L$(OPENSHMEM_INSTALL)/lib -lsma
SOS_LD_FLAGS=-L$(OPENSHMEM_INSTALL)/lib64 -L$(OPENSHMEM_INSTALL)/lib -lsma

all: $(TARGET)

bfs_oshmem: main.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3

bfs_oshmem-mailbox: main-mailbox.c utilities.c mrg.c generator.c splittable_mrg.c
	gcc -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox: main-single-mailbox.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-hiper: main-single-mailbox-hiper.c utilities.c mrg.c splittable_mrg.c generator.c
	g++ -std=c++11 -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE -I$(HCLIB_ROOT)/include -I$(HCLIB_HOME)/modules/system/inc -I$(HCLIB_HOME)/modules/openshmem/inc -L$(HCLIB_ROOT)/lib -L$(HCLIB_HOME)/modules/openshmem/lib -lhclib -lhclib_openshmem

bfs_oshmem-single-mailbox-concurrent: main-single-mailbox-concurrent.c utilities.c mrg.c splittable_mrg.c generator.c
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c99 $(FLAGS) -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-concurrent-ctx: main-single-mailbox-concurrent-ctx.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -std=c99 $(FLAGS) -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-bitvector: main-bitvector.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -std=c99 $(FLAGS) -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-concurrent-crc: main-single-mailbox-concurrent-crc.cpp crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c++11 $(FLAGS) -DUSE_CITY64 -DCRC32 -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -Icrc -Ismhasher/src -Icityhash/src -Icityhash -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-concurrent-crc-hiper: main-single-mailbox-concurrent-crc-hiper.c crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	g++ -std=c++11 $(FLAGS) -DUSE_CITY64 -DCRC32 -Icrc -Ismhasher/src -Icityhash/src -Icityhash -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE -I$(HCLIB_ROOT)/include -I$(HCLIB_HOME)/modules/system/inc -I$(HCLIB_HOME)/modules/openshmem/inc -L$(HCLIB_ROOT)/lib -L$(HCLIB_HOME)/modules/openshmem/lib -lhclib -lhclib_openshmem

bfs_oshmem-single-mailbox-concurrent-crc-hiper-contexts: main-single-mailbox-concurrent-crc-hiper-contexts.c crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	g++ -std=c++11 $(FLAGS) -DUSE_CITY64 -DCRC32 -Icrc -Ismhasher/src -Icityhash/src -Icityhash -I$(OPENSHMEM_INSTALL)/include $(SOS_LD_FLAGS) -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE -I$(HCLIB_ROOT)/include -I$(HCLIB_HOME)/modules/system/inc -I$(HCLIB_HOME)/modules/sos/inc -L$(HCLIB_ROOT)/lib -L$(HCLIB_HOME)/modules/sos/lib -lhclib -lhclib_sos


bfs_oshmem-single-mailbox-strided: main-single-mailbox-strided.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-replicated: main-single-mailbox-replicated.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-omp: main-single-mailbox-omp.c utilities.c mrg.c splittable_mrg.c generator.c
	gcc -fopenmp -I$(OPENSHMEM_INSTALL)/include $(OSHMEM_LD_FLAGS) -lrt -o $@ $^ -O3 -DGENERATOR_USE_PACKED_EDGE_TYPE

bfs_oshmem-single-mailbox-atomics: main-single-mailbox-atomics.c crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c99 $(FLAGS) -DUSE_CITY64 -DCRC32 -Icrc -Ismhasher/src -Icityhash/src -Icityhash -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE -fopenmp

bfs_oshmem-single-mailbox-atomics-runtime-safety: main-single-mailbox-atomics.c crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	$(OPENSHMEM_INSTALL)/bin/oshcc $(FLAGS) -DUSE_CITY64 -DCRC32 -Icrc -Ismhasher/src -Icityhash/src -Icityhash -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE -fopenmp -DRUNTIME_SAFETY


bfs_oshmem-single-mailbox-wavefront: main-single-mailbox-wavefront.c crc/crc.c smhasher/src/MurmurHash3.cpp cityhash/src/city.cc utilities.c mrg.c splittable_mrg.c generator.c
	$(OPENSHMEM_INSTALL)/bin/oshcc -std=c99 $(FLAGS) -DUSE_CITY64 -DCRC32 -Icrc -Ismhasher/src -Icityhash/src -Icityhash -lrt -o $@ $^ -DGENERATOR_USE_PACKED_EDGE_TYPE -fopenmp

clean:
	rm -f $(TARGET)
