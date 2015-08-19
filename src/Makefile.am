SUBDIRS =

# Includes folders
INCLUDES_DIR = -I$(top_srcdir)/inc -I$(top_srcdir)/src/inc

# cflags: important to define that otherwise we inherit default values too
CXXFLAGS = -Wall -g -O3 -std=c++11
LDFLAGS = -lpthread

if DIST_WS
HUPCPP_FLAGS_1 = -DDIST_WS
else
HUPCPP_FLAGS_1 =
endif

if HUPCPP
HUPCPP_FLAGS_2 = -DHUPCPP
else
HUPCPP_FLAGS_2 =
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

if WITH_OCR
AM_CXXFLAGS = -DWITH_OCR -I$(HCLIB_ROOT)/include
libhcpp_la_SOURCES = ocr-runtime.cpp
else
AM_CXXFLAGS = $(HUPCPP_FLAGS_1) $(HUPCPP_FLAGS_2) $(PRODUCTION_SETTINGS_FLAGS) -I$(LIBXML2)/include/libxml2
libhcpp_la_SOURCES = hcpp-runtime.cpp hcpp-deque.cpp hcpp-mm.cpp hcpp-hpt.cpp hcpp-thread-bind.cpp hcpp-ddf.cpp
endif

# build hcpp
libhcpp_la_CXXFLAGS= $(CXXFLAGS) $(INCLUDES_DIR) $(AM_CXXFLAGS)

MAINTAINERCLEANFILES = Makefile.in
