include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak
# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

HCLIB_PERF_CC?=icc
C_C = $(HCLIB_PERF_CC)

OMP_FLAG =

# link objects (binaries) together
a.out:		main.o \
			kernel_cpu.o \
			num.o \
			timer.o
	$(C_C)	main.o \
			kernel_cpu.o \
			num.o \
			timer.o \
			-lm \
			$(OMP_FLAG) \
			-o lavaMD $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)
# compile function files into objects (binaries)
main.o:		main.h \
			main.c \
			kernel_cpu.h \
			kernel_cpu.c \
			num.h \
			num.c \
			timer.h \
			timer.c
	$(C_C)	${OUTPUT} main.c \
			-c \
			-o main.o \
			-O3 $(HCLIB_CXXFLAGS)

kernel_cpu.o:	kernel_cpu.h \
						kernel_cpu.c
	$(C_C)				kernel_cpu.c \
						-c \
						-o kernel_cpu.o \
						-O3 \
						$(OMP_FLAG) $(HCLIB_CXXFLAGS)

num.o:	num.h \
					num.c
	$(C_C)			num.c \
					-c \
					-o num.o \
					-O3 $(HCLIB_CXXFLAGS)

timer.o:	timer.h \
						timer.c
	$(C_C)				timer.c \
						-c \
						-o timer.o \
						-O3 $(HCLIB_CXXFLAGS)

# delete all object and executable files
clean:
	rm	-f *.o \
		lavaMD
