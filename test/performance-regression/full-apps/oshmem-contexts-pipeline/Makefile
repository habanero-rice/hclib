all:shmem_ctx_pipelined_reduce shmem_ctx_pipelined_reduce_no_context

shmem_ctx_pipelined_reduce: shmem_ctx_pipelined_reduce.c
	$(OPENSHMEM_INSTALL)/bin/oshcc $^ -o $@ -O2

shmem_ctx_pipelined_reduce_no_context: shmem_ctx_pipelined_reduce.c
	$(OPENSHMEM_INSTALL)/bin/oshcc -DNO_CONTEXTS $^ -o $@ -O2

clean:
	rm -f shmem_ctx_pipelined_reduce
