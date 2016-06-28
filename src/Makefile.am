SUBDIRS =

# Includes folders
INCLUDES_DIR = -I$(top_srcdir)/inc -I$(top_srcdir)/src/inc -I$(top_srcdir)/src/fcontext

# cflags: important to define that otherwise we inherit default values too
CFLAGS = -Wall -g -O3 -std=c99
CXXFLAGS = -Wall -g -O3 -std=c++11
LDFLAGS = -lpthread

if DIST_WS
HC_FLAGS_1 = -DDIST_WS
else
HC_FLAGS_1 =
endif

if HCSHMEM
HC_FLAGS_2 = -DHCSHMEM -DHC_COMM_WORKER
else
HC_FLAGS_2 =
endif

if HUPCPP
HC_FLAGS_2 = -DHUPCPP -DHC_COMM_WORKER
else
HC_FLAGS_2 =
endif

if HC_CUDA
HC_FLAGS_3 = -DHC_CUDA
INCLUDES_DIR += -I${CUDA_HOME}/include
else
HC_FLAGS_3 = 
endif

if HC_VERBOSE
HC_FLAGS_4 = -DVERBOSE
else
HC_FLAGS_4 =
endif

if PRODUCTION_SETTINGS
PRODUCTION_SETTINGS_FLAGS =
else
PRODUCTION_SETTINGS_FLAGS = -DHC_ASSERTION_CHECK -DHC_COMM_WORKER_STATS
endif


# build the hclib library
lib_LTLIBRARIES = libhclib.la

# These are populated in sub-folder's Makefile.inc
noinst_LTLIBRARIES =
libhclib_la_LIBADD =

AM_CXXFLAGS = $(HC_FLAGS_1) $(HC_FLAGS_2) $(HC_FLAGS_3) $(HC_FLAGS_4) \
			  $(PRODUCTION_SETTINGS_FLAGS) -I$(LIBXML2_INCLUDE)
libhclib_la_SOURCES = hclib-runtime.c hclib-deque.c hclib-hpt.c hclib-thread-bind.c \
					 hcshmem-support.c hclib-promise.c hclib-timer.c hcupc-support.c hclib_cpp.cpp hclib.c hclib-tree.c hclib-isolated.c hashmap.c

if X86
if OSX
libhclib_la_SOURCES += \
fcontext/jump_i386_sysv_macho_gas.S \
fcontext/make_i386_sysv_macho_gas.S
else
if LINUX
libhclib_la_SOURCES += \
fcontext/jump_i386_sysv_elf_gas.S \
fcontext/make_i386_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
endif # OSX
else
if X86_64
if OSX
libhclib_la_SOURCES += \
fcontext/jump_x86_64_sysv_macho_gas.S \
fcontext/make_x86_64_sysv_macho_gas.S
else
if LINUX
libhclib_la_SOURCES += \
fcontext/jump_x86_64_sysv_elf_gas.S \
fcontext/make_x86_64_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
endif # OSX
else
$(error Unknown architecture)
endif # X86_64
endif # X86

# build hclib
libhclib_la_CXXFLAGS= $(CXXFLAGS) $(INCLUDES_DIR) $(AM_CXXFLAGS)
libhclib_la_CFLAGS= $(CFLAGS) $(INCLUDES_DIR) $(AM_CXXFLAGS)

MAINTAINERCLEANFILES = Makefile.in
