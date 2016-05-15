include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler

HCLIB_PERF_CC?=icc
CC = $(HCLIB_PERF_CC)
CC_FLAGS = -g -fopenmp -O0

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans $(HCLIB_CXXFLAGS) $(HCLIB_LDFLAGS) $(HCLIB_LDLIBS)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c $(HCLIB_CXXFLAGS)

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c $(HCLIB_CXXFLAGS)
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c $(HCLIB_CXXFLAGS)
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c $(HCLIB_CXXFLAGS)

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c $(HCLIB_CXXFLAGS)

clean:
	rm -f *.o *~ kmeans 
