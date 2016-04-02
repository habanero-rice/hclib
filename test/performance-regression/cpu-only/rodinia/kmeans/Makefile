include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.pre.mak
include $(HCLIB_ROOT)/include/hclib.mak
include $(HCLIB_ROOT)/../modules/system/inc/hclib_system.post.mak
# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	g++ $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c $(PROJECT_CXXFLAGS)

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c $(PROJECT_CXXFLAGS)
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c $(PROJECT_CXXFLAGS)
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c $(PROJECT_CXXFLAGS)

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c $(PROJECT_CXXFLAGS)

clean:
	rm -f *.o *~ kmeans 
