include $(HCLIB_ROOT)/include/hclib.mak
all: avimod.o avilib.o

# compile supporting function file into object (binary)
avimod.o: avilib.h avimod.c 
	gcc -c avimod.c $(PROJECT_CXXFLAGS)

# compile supporting function file into object (binary)
avilib.o: avilib.h avilib.c
	gcc -c avilib.c $(PROJECT_CXXFLAGS)

# delete files
clean:
	rm *.o *.out
