include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
include ../../custom.mak

HCLIB_PERF_CXX?=icpc
OMP_NUM_THREADS = 8

euler3d_cpu_double: euler3d_cpu_double.o
	$(HCLIB_PERF_CXX) -O3 -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.o -o euler3d_cpu_double $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

euler3d_cpu_double.o: euler3d_cpu_double.cpp
	$(HCLIB_PERF_CXX) -c -O3 -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double.o $(HCLIB_CXXFLAGS)

clean:
	rm -f euler3d_cpu_double *.o
