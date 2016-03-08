include $(HCLIB_ROOT)/include/hclib.mak

OMP_NUM_THREADS = 8

euler3d_cpu_double: euler3d_cpu_double.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

clean:
	rm -f euler3d_cpu_double
