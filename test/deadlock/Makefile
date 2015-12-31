TARGET := deadlock

$(TARGET): $(TARGET).c
	@ [ -d "${HCLIB_ROOT}" ] || (echo Need to set HCLIB_ROOT; exit 1)
	gcc $^ -o$@ -pthread -std=gnu99 -I${HCLIB_ROOT}/include ${HCLIB_ROOT}/lib/libhclib.a -lxml2 -L${LIBXML2_LIBS}

NPROC ?= 4

.PHONY: run
run: $(TARGET)
	./$(TARGET) -nproc $(NPROC)

clean:
	rm -f $(TARGET)

