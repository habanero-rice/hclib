# check required autoconf version
AC_PREREQ([2.59])

# define distribution properties
AC_INIT([hclib], [0.1], [vivekk@rice.edu])

# check if srcdir is correct
AC_CONFIG_SRCDIR([inc/hclib.h])

AC_CONFIG_AUX_DIR([config])

AM_INIT_AUTOMAKE([1.9 tar-ustar no-define foreign dist-bzip2])

AC_COPYRIGHT([])

# configure script revision
AC_REVISION([$Revision: 0.1 $])

AM_PROG_AS
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_LANG(C++)
AC_CANONICAL_HOST

# Platform check
AM_CONDITIONAL([OSX], [case $host_os in darwin*) true;; *) false;; esac])
AM_CONDITIONAL([LINUX], [case $host_os in *linux*) true;; *) false;; esac])
AM_CONDITIONAL([X86], [case $host_cpu in i?86) true;; *) false;; esac])
AM_CONDITIONAL([X86_64], [case $host_cpu in x86_64) true;; *) false;; esac])
AC_MSG_NOTICE([Host detected as $host_os $host_cpu])

CPPFLAGS=$CPPFLAGS_BACKUP

###################################################################
########### GET LOW LEVEL DETAILS IN THE INSTALLATION #############
###################################################################

### ENABLE DISTRIBUTED WORK_STEALING SUPPORT FOR HabaneroUPC++
AC_ARG_ENABLE(distWS,
    AS_HELP_STRING([--enable-distWS],
    [enable dist. W.S. support (Default is false)]),
    [with_distWS=$enableval],
    [with_distWS=no;])

AS_IF([test "x$with_distWS" != xno],
      [ AC_MSG_NOTICE([Enabled Dist. W.S. support]) ],
      [ AC_MSG_NOTICE([Disabled Dist. W.S. support]) ])

AM_CONDITIONAL(DIST_WS, test "x$with_distWS" != xno)
### End Dist WS

### SUPPORT HabaneroOpenSHMEM++
AC_ARG_ENABLE(hcshmem,
    AS_HELP_STRING([--enable-hcshmem],
    [support HabaneroOpenSHMEM++ runtime (Default is false)]),
    [with_hcshmem=$enableval],
    [with_hcshmem=no;])

AS_IF([test "x$with_hcshmem" != xno],
      [ AC_MSG_NOTICE([Enabled HabaneroOpenSHMEM++ support]) ],
      [ AC_MSG_NOTICE([Disabled HabaneroOpenSHMEM++ support]) ])

AM_CONDITIONAL(HCSHMEM, test "x$with_hcshmem" != xno)
### End hupcpp

### SUPPORT HabaneroUPC++
AC_ARG_ENABLE(hupcpp,
    AS_HELP_STRING([--enable-hupcpp],
    [support HabaneroUPC++ runtime (Default is false)]),
    [with_hupcpp=$enableval],
    [with_hupcpp=no;])

AS_IF([test "x$with_hupcpp" != xno],
      [ AC_MSG_NOTICE([Enabled HabaneroUPC++ support]) ],
      [ AC_MSG_NOTICE([Disabled HabaneroUPC++ support]) ])

AM_CONDITIONAL(HUPCPP, test "x$with_hupcpp" != xno)
### End hupcpp

### SUPPORT CUDA-accelerated forasyncs
AC_ARG_ENABLE(cuda,
    AS_HELP_STRING([--enable-cuda],
    [support CUDA-accelerated forasyncs (Default is false)]),
    [with_cuda=$enableval],
    [with_cuda=no;])

AS_IF([test "x$with_cuda" != xno],
      [ AC_MSG_NOTICE([Enabled CUDA support]) ],
      [ AC_MSG_NOTICE([Disabled CUDA support]) ])

AM_CONDITIONAL(HC_CUDA, test "x$with_cuda" != xno)
### End cuda

### Turn on runtime verbosity
AC_ARG_ENABLE(verbose,
    AS_HELP_STRING([--enable-verbose],
    [turn on HC runtime verbosity (Default is false)]),
    [with_verbose=$enableval],
    [with_verbose=no;])

AS_IF([test "x$with_verbose" != xno],
      [ AC_MSG_NOTICE([Enabled verbose HC runtime]) ],
      [ AC_MSG_NOTICE([Disabled verbose HC runtime]) ])

AM_CONDITIONAL(HC_VERBOSE, test "x$with_verbose" != xno)
### End verbose

### ENABLE PRODUCTION SETTINGS
AC_ARG_ENABLE(production,
    AS_HELP_STRING([--enable-production],
    [disable assertion and statistics (Default is true)]),
    [with_production=$enableval],
    [with_production=no;])

AS_IF([test "x$with_production" != xno],
      [ AC_MSG_NOTICE([Production settings enabled]) ],
      [ AC_MSG_NOTICE([Production settings disabled]) ])

AM_CONDITIONAL(PRODUCTION_SETTINGS, test "x$with_production" != xno)
### End PRODUCTION settings

### C11 features check
AC_LANG_PUSH(C)
_old_CFLAGS="${CFLAGS}"
CFLAGS="${_old_CFLAGS} -std=c11"
# We require basic C11 support
AC_CACHE_CHECK([for basic C11 support], [ac_cv_c11_lang],
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [])],
                   [ac_cv_c11_lang=yes],
                   [ac_cv_c11_lang=no])])
if test $ac_cv_c11_lang = no; then
    AC_MSG_ERROR(C11 support (-std=c11) is requred)
fi
# NOTE: Sadly, we can't trust __STDC_NO_ATOMICS__
AC_CACHE_CHECK([for C11 stdatomic.h support], [ac_cv_c11_stdatomic],
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include  <stdatomic.h>], [_Atomic int foo;])],
                   [ac_cv_c11_stdatomic=yes],
                   [ac_cv_c11_stdatomic=no])])
if test $ac_cv_c11_stdatomic = yes; then
    AC_DEFINE([HAVE_C11_STDATOMIC], [1], [Defined if C11 stdatomic.h is available.])
fi
CFLAGS="${_old_CFLAGS}"
AC_LANG_POP(C)
### End C11 features check

### C++11 features check
_old_CXXFLAGS="${CXXFLAGS}"
CXXFLAGS="${_old_CXXFLAGS} -std=c++11"
# We require basic C++11 support
AC_CACHE_CHECK([for basic C++11 support], [ac_cv_cxx11_lang],
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [])],
                   [ac_cv_cxx11_lang=yes],
                   [ac_cv_cxx11_lang=no])])
if test $ac_cv_cxx11_lang = no; then
    AC_MSG_ERROR(C++11 support (-std=c++11) is requred)
fi
AC_CACHE_CHECK([for C++11 std::is_trivially_copyable support], [ac_cv_cxx11_trivial_copy_check],
[AC_COMPILE_IFELSE([AC_LANG_PROGRAM([#include  <type_traits>], [return std::is_trivially_copyable<int>::value;])],
                   [ac_cv_cxx11_trivial_copy_check=yes],
                   [ac_cv_cxx11_trivial_copy_check=no])])
if test $ac_cv_cxx11_trivial_copy_check = yes; then
    AC_DEFINE([HAVE_CXX11_TRIVIAL_COPY_CHECK], [1], [Defined if C++11 std::is_trivially_copyable is supported.])
fi
CXXFLAGS="${_old_CXXFLAGS}"
### End C++11 features check

# Variable substitutions.
AC_SUBST([ac_aux_dir])
AC_SUBST([abs_top_srcdir])

#
# Config file to process
#
AC_CONFIG_FILES([
Makefile
src/Makefile
])

# Header files
AC_CONFIG_HEADERS([inc/hclib_config.h])

# Generates and runs config.status, which in turn creates the makefiles 
# and any other files resulting from configuration
AC_OUTPUT
