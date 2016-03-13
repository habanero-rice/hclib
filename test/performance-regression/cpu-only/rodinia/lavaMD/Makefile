include $(HCLIB_ROOT)/include/hclib.mak
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

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

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
			$(OMP_LIB) \
			-o lavaMD $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)
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
			-O3 $(PROJECT_CXXFLAGS)

kernel_cpu.o:	kernel_cpu.h \
						kernel_cpu.c
	$(C_C)				kernel_cpu.c \
						-c \
						-o kernel_cpu.o \
						-O3 \
						$(OMP_FLAG) $(PROJECT_CXXFLAGS)

num.o:	num.h \
					num.c
	$(C_C)			num.c \
					-c \
					-o num.o \
					-O3 $(PROJECT_CXXFLAGS)

timer.o:	timer.h \
						timer.c
	$(C_C)				timer.c \
						-c \
						-o timer.o \
						-O3 $(PROJECT_CXXFLAGS)

# delete all object and executable files
clean:
	rm	-f *.o \
		lavaMD
