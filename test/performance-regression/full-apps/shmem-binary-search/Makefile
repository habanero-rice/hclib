TARGETS=binary-search binary-search-thread-init binary-search-pthreads binary-search-contexts
FLAGS=-O2

all: $(TARGETS)

binary-search: binary-search.c
	 $(SOS_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^

binary-search-thread-init: binary-search.c
	$(SOS_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_THREAD_INITIALIZER

binary-search-pthreads: binary-search.c
	$(SOS_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_THREAD_INITIALIZER -DUSE_PTHREADS

binary-search-contexts: binary-search.c
	$(SOS_INSTALL)/bin/oshcc -o $@ $(FLAGS) $^ -DUSE_THREAD_INITIALIZER -DUSE_PTHREADS -DUSE_CONTEXTS

clean:
	rm -f $(TARGETS)
