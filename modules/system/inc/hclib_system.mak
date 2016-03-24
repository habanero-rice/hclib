PROJECT_CFLAGS+=-I$(HCLIB_ROOT)/../modules/system/inc
PROJECT_CXXFLAGS+=-I$(HCLIB_ROOT)/../modules/system/inc
PROJECT_LDFLAGS+=-L$(HCLIB_ROOT)/../modules/system/lib
PROJECT_LDLIBS+=-Wl,--whole-archive -lhclib_system -Wl,--no-whole-archive
