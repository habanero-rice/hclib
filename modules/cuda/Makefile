include $(HCLIB_ROOT)/include/hclib.mak

OBJS=obj/hclib_cuda.o

lib/libhclib_cuda.so: $(OBJS)
	g++ -shared $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS) -o $@ $(OBJS)

obj/hclib_cuda.o: src/hclib_cuda.cpp
	g++ -Iinc $^ -o $@ $(PROJECT_CFLAGS) --compile -fPIC

clean:
	rm -f obj/*.o lib/*
