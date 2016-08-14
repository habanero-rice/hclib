TARGET := fib

include $(HCLIB_ROOT)/include/hclib.mak

$(TARGET): $(TARGET).c
	gcc $^ -o$@ -std=gnu11 $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

WORKLOAD_ARGS ?= 5

NPROC ?= 4

.PHONY: run
run: $(TARGET)
	$(SETUP_ENV) HCLIB_WORKERS=$(NPROC) ./$(TARGET) $(WORKLOAD_ARGS)

.PHONY: hoard
hoard: SETUP_ENV=LD_PRELOAD="/opt/local/Hoard/lib/libhoard.so"
hoard: run

.PHONY: tbbmalloc
tbbmalloc: SETUP_ENV=LD_PRELOAD="/usr/lib/libtbbmalloc_proxy.so"
tbbmalloc: run

.PHONY: tcmalloc
tcmalloc: SETUP_ENV=LD_PRELOAD="/usr/lib/libtcmalloc.so.4"
tcmalloc: run

.PHONY: jemalloc
jemalloc: SETUP_ENV=LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so"
jemalloc: run

clean:
	rm -f $(TARGET)
