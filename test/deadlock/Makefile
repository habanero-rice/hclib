TARGET := deadlock

include $(HCLIB_ROOT)/include/hclib.mak

$(TARGET): $(TARGET).c
	$(CC) $^ -o$@ -std=gnu11 $(PROJECT_CFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

NPROC ?= 4

.PHONY: run
run: $(TARGET)
	HCLIB_WORKERS=$(NPROC) ./$(TARGET)

clean:
	rm -f $(TARGET)

