TARGET := deadlock

$(TARGET): $(TARGET).c
	@ [ -d "${HCPP_ROOT}" ] || (echo Need to set HCPP_ROOT; exit 1)
	gcc $^ -o$@ -pthread -std=gnu99 -I${HCPP_ROOT}/include ${HCPP_ROOT}/lib/libhcpp.a -lxml2 -L${LIBXML2_LIBS}

NPROC ?= 4

.PHONY: run
run: $(TARGET)
	./$(TARGET) -nproc $(NPROC)

clean:
	rm -f $(TARGET)

