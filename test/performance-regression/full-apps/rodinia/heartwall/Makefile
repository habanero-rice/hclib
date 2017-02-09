include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

HCLIB_PERF_CC?=icc
CC = $(HCLIB_PERF_CC)

# link objects(binaries) together
heartwall: main.o kernel.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) main.o kernel.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

# compile main function file into object (binary)
main.o: main.c
	$(CC) $(OUTPUT) main.c -I./AVI -c -O3 $(HCLIB_CXXFLAGS)

kernel.o: kernel.c
	$(CC) $(OUTPUT) kernel.c -I./AVI -c -O3 $(HCLIB_CXXFLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall
