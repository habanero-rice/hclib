all: hwloc_visualizer hwloc_to_hpt

hwloc_visualizer: hwloc_visualizer.cpp
	g++ hwloc_visualizer.cpp common.cpp -o $@ -I${HWLOC_HOME}/include -L${HWLOC_HOME}/lib -lhwloc -O0 -g

hwloc_to_hpt: hwloc_to_hpt.cpp
	g++ hwloc_to_hpt.cpp common.cpp -o $@ -I${HWLOC_HOME}/include -L${HWLOC_HOME}/lib -lhwloc -O0 -g

clean:
	rm -f hwloc_visualizer hwloc_to_hpt
