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

if HUPCPP
HC_FLAGS_2 = -DHUPCPP -DHC_COMM_WORKER
else
HC_FLAGS_2 =
endif

if HC_CUDA
HC_FLAGS_3 = -DHC_CUDA
INCLUDES_DIR += ${CUDA_HOME}/include
else
HC_FLAGS_3 = 
endif

if PRODUCTION_SETTINGS
PRODUCTION_SETTINGS_FLAGS =
else
PRODUCTION_SETTINGS_FLAGS = -DHC_ASSERTION_CHECK -DHC_COMM_WORKER_STATS
endif


# build the hcpp library
lib_LTLIBRARIES = libhcpp.la

# These are populated in sub-folder's Makefile.inc
noinst_LTLIBRARIES =
libhcpp_la_LIBADD =

AM_CXXFLAGS = $(HC_FLAGS_1) $(HC_FLAGS_2) $(HC_FLAGS_3) \
			  $(PRODUCTION_SETTINGS_FLAGS) -I$(LIBXML2_INCLUDE)
libhcpp_la_SOURCES = hcpp-runtime.c hcpp-deque.c hcpp-hpt.c hcpp-thread-bind.c \
					 hcpp-ddf.c hcpp-timer.c hcupc-support.c hclib_cpp.cpp hclib.c

if X86
if OSX
libhcpp_la_SOURCES += \
fcontext/jump_i386_sysv_macho_gas.S \
fcontext/make_i386_sysv_macho_gas.S
else
if LINUX
libhcpp_la_SOURCES += \
fcontext/jump_i386_sysv_elf_gas.S \
fcontext/make_i386_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
endif # OSX
else
if X86_64
if OSX
libhcpp_la_SOURCES += \
fcontext/jump_x86_64_sysv_macho_gas.S \
fcontext/make_x86_64_sysv_macho_gas.S
else
if LINUX
libhcpp_la_SOURCES += \
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

# build hcpp
libhcpp_la_CXXFLAGS= $(CXXFLAGS) $(INCLUDES_DIR) $(AM_CXXFLAGS)
libhcpp_la_CFLAGS= $(CFLAGS) $(INCLUDES_DIR) $(AM_CXXFLAGS)

MAINTAINERCLEANFILES = Makefile.in
