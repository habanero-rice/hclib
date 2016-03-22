include $(HCLIB_ROOT)/include/hclib.mak
CC = g++
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = -fopenmp $(PROJECT_CXXFLAGS) $(PROJECT_LDFLAGS) $(PROJECT_LDLIBS)

release:
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CC) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder


