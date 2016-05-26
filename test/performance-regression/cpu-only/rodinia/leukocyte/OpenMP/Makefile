include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../../custom.mak
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

HCLIB_PERF_CC?=icc
CC = $(HCLIB_PERF_CC)

# C compiler
CC_FLAGS = -g -O3 -Wall -fopenmp -I$(MATRIX_DIR) $(HCLIB_CXXFLAGS)

# Matrix library path
MATRIX_DIR = ../meschach_lib


leukocyte: detect_main.o avilib.o find_ellipse.o track_ellipse.o misc_math.o $(MATRIX_DIR)/meschach.a
	$(CC) $(CC_FLAGS) -lm avilib.o find_ellipse.o track_ellipse.o misc_math.o detect_main.o -o leukocyte -lm $(MATRIX_DIR)/meschach.a -lpthread $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

%.o: %.c
	$(CC) $(OUTPUT) $(CC_FLAGS) $< -c

detect_main.o: find_ellipse.h track_ellipse.h avilib.h
find_ellipse.o: avilib.h
track_ellipse.o: find_ellipse.h track_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o leukocyte
