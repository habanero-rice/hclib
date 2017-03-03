SUBDIRS =

# Includes folders
INCLUDES_DIR = -I$(top_srcdir)/inc -I$(top_srcdir)/src/inc -I$(top_srcdir)/src/fcontext -I$(JSMN_HOME)

# cflags: important to define that otherwise we inherit default values too
CFLAGS = -Wall -g -O3 -std=c11
CXXFLAGS = -Wall -g -O3 -std=c++11

LDFLAGS = -lpthread

if HC_STATS
HC_FLAGS_STATS = -DHCLIB_STATS
else
HC_FLAGS_STATS =
endif

if HC_VERBOSE
HC_FLAGS_VERBOSE = -DVERBOSE
else
HC_FLAGS_VERBOSE =
endif

if PRODUCTION_SETTINGS
PRODUCTION_SETTINGS_FLAGS =
else
PRODUCTION_SETTINGS_FLAGS = -DHC_ASSERTION_CHECK
endif

# build the hclib library
lib_LTLIBRARIES = libhclib.la

# These are populated in sub-folder's Makefile.inc
noinst_LTLIBRARIES =
libhclib_la_LIBADD =

AM_CXXFLAGS = $(HC_FLAGS_1) $(HC_FLAGS_2) $(HC_FLAGS_3) $(HC_FLAGS_4) \
			  $(PRODUCTION_SETTINGS_FLAGS) $(shell xml2-config --cflags)
libhclib_la_SOURCES = hclib-runtime.c hclib-deque.c hclib-promise.c \
					  hclib-timer.c hclib_cpp.cpp hclib.c hclib-tree.c hclib-locality-graph.c \
					  hclib_module.cpp hclib-fptr-list.c hclib-mem.c hclib-instrument.c \
					  hclib_atomic.c

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
