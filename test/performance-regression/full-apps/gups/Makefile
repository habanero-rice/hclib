TARGETS=gups gups-xor
FLAGS=-O0 -g

all: $(TARGETS)

gups: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^

gups-xor: gups.c
	 $(OPENSHMEM_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_BXOR_ATOMICS

clean:
	rm -f $(TARGETS)
