all: hwloc_visualizer hwloc_to_hpt hclib_instrument_parser

hclib_instrument_parser: hclib_instrument_parser.c
	gcc $^ -o $@ -I../hclib-install/include

hwloc_visualizer: hwloc_visualizer.cpp
	$(CXX) hwloc_visualizer.cpp common.cpp -o $@ -I${HWLOC_HOME}/include -L${HWLOC_HOME}/lib -lhwloc -O0 -g

hwloc_to_hpt: hwloc_to_hpt.cpp
	$(CXX) hwloc_to_hpt.cpp common.cpp -o $@ -I${HWLOC_HOME}/include -L${HWLOC_HOME}/lib -lhwloc -O0 -g

clean:
	rm -f hwloc_visualizer hwloc_to_hpt
