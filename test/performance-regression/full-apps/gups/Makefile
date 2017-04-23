TARGETS=gups gups-xor gups-xor-pthreads gups-xor-pthreads-contexts
FLAGS=-O0 -g

all: $(TARGETS)

gups: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^

gups-xor: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_BXOR_ATOMICS

gups-xor-pthreads: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_BXOR_ATOMICS -DUSE_PTHREADS

gups-xor-pthreads-contexts: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_BXOR_ATOMICS -DUSE_PTHREADS -DUSE_CONTEXTS

clean:
	rm -f $(TARGETS)
