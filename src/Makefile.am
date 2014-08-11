SUBDIRS =

# Includes folder we need when compiling

HABANEROCPP_INCLUDES_DIR = -I$(top_srcdir)/inc -I$(top_srcdir)/src -I$(HCLIB_ROOT)/include

# cflags: important to define that otherwise we inherit default values too
CXXFLAGS = -g -Wall

HABANEROCPP_CXXFLAGS = "-std=c++11"

# Standard am options all the submodules can grab
HABANEROCPP_AM_CXXFLAGS = $(HABANEROCPP_INCLUDES_DIR) $(HABANEROCPP_CXXFLAGS)

# Add warning treated as errors for build
# For modules that do need to bypass this, just use the
AM_CXXFLAGS = $(HABANEROCPP_AM_CXXFLAGS)

lib_LTLIBRARIES = libhcpp.la

# build the runtime library
libhcpp_la_SOURCES = hcpp-runtime.cpp asyncAwait.cpp asyncPhased.cpp

libhcpp_la_CXXFLAGS = $(AM_CXXFLAGS)

MAINTAINERCLEANFILES = Makefile.in
