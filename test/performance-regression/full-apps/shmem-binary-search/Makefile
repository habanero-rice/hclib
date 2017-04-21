all: binary-search binary-search-thread-init

binary-search: binary-search.c
	 $(SOS_INSTALL)/bin/oshcc -o $@ -O2 $^

binary-search-thread-init: binary-search.c
	$(SOS_INSTALL)/bin/oshcc -o $@ -O2 $^ -DUSE_THREAD_INITIALIZER

clean:
	rm -f binary-search binary-search-thread-init
