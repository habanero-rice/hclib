SUBDIRS =

# Includes folders
INCLUDES_DIR = -I$(top_srcdir)/inc -I$(top_srcdir)/src/inc -I$(top_srcdir)/src/fcontext

HC_DEFS :=

if HC_VERBOSE
HC_DEFS += -DVERBOSE
endif

if HC_RUNTIME_STATS
HC_DEFS += -DHC_COMM_WORKER_STATS
endif

if HAS_HCLIB_WORKER_STRATEGY
HC_DEFS += -DHCLIB_SET_WORKER_STRATEGY=@HCLIB_SET_WORKER_STRATEGY@
endif

if HAS_HCLIB_WORKER_OPTIONS
HC_DEFS += -DHCLIB_SET_WORKER_OPTIONS=@HCLIB_SET_WORKER_OPTIONS@
endif

# build the hclib library
lib_LTLIBRARIES = libhclib.la libhclib_dbg.la

# These are populated in sub-folder's Makefile.inc
noinst_LTLIBRARIES =

libhclib_common_SOURCES = hclib-runtime.c hclib-deque.c hclib-hpt.c hclib-thread-bind.c \
					 hclib-promise.c hclib-timer.c hclib_cpp.cpp hclib.c hclib-tree.c

if X86
if OSX
libhclib_common_SOURCES += \
fcontext/jump_i386_sysv_macho_gas.S \
fcontext/make_i386_sysv_macho_gas.S
else
if LINUX
libhclib_common_SOURCES += \
fcontext/jump_i386_sysv_elf_gas.S \
fcontext/make_i386_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
endif # OSX
else
if X86_64
if OSX
libhclib_common_SOURCES += \
fcontext/jump_x86_64_sysv_macho_gas.S \
fcontext/make_x86_64_sysv_macho_gas.S
else
if LINUX
libhclib_common_SOURCES += \
fcontext/jump_x86_64_sysv_elf_gas.S \
fcontext/make_x86_64_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
endif # OSX
else
if PPC64
if LINUX
libhclib_common_SOURCES += \
fcontext/jump_ppc64_sysv_elf_gas.S \
fcontext/make_ppc64_sysv_elf_gas.S
else
$(error Unknown OS)
endif # LINUX
else
$(error Unknown architecture)
endif # PPC64
endif # X86_64
endif # X86

CFLAGS_COMMON     := -Wall -pthread $(shell xml2-config --cflags) \
                     $(INCLUDES_DIR) $(HC_DEFS)
CFLAGS_PRODUCTION := -O3 -DNDEBUG
CFLAGS_DEBUG      := -O@dbg_flag_lvl@ -g -DHCLIB_DEBUG -DHC_ASSERTION_CHECK

# build hclib
libhclib_common_LIBADD   =
libhclib_common_CFLAGS   = -std=c11 $(CFLAGS_COMMON)
libhclib_common_CXXFLAGS = -std=c++11 $(CFLAGS_COMMON)
libhclib_common_LDFLAGS  =

MAINTAINERCLEANFILES = Makefile.in

# Debug build
libhclib_dbg_la_LIBADD   = $(libhclib_common_LIBADD)
libhclib_dbg_la_SOURCES  = $(libhclib_common_SOURCES)
libhclib_dbg_la_CFLAGS   = $(CFLAGS_DEBUG) $(libhclib_common_CFLAGS)
libhclib_dbg_la_CXXFLAGS = $(CFLAGS_DEBUG) $(libhclib_common_CXXFLAGS)
libhclib_dbg_la_LDFLAGS  = $(libhclib_common_LDFLAGS)

# Production build
libhclib_la_LIBADD   = $(libhclib_common_LIBADD)
libhclib_la_SOURCES  = $(libhclib_common_SOURCES)
libhclib_la_CFLAGS   = $(CFLAGS_PRODUCTION) $(libhclib_common_CFLAGS)
libhclib_la_CXXFLAGS = $(CFLAGS_PRODUCTION) $(libhclib_common_CXXFLAGS)
libhclib_la_LDFLAGS  = $(libhclib_common_LDFLAGS)

# cflags: important to define that otherwise we inherit default values too
CFLAGS   =
CXXFLAGS =
LDFLAGS  =
