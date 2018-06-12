# check required autoconf version
AC_PREREQ([2.59])

# define distribution properties
AC_INIT([hclib], [0.1], [jmg3@rice.edu])

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
AM_CONDITIONAL([PPC64], [case $host_cpu in powerpc64le) true;; *) false;; esac])
AC_MSG_NOTICE([Host detected as $host_os $host_cpu])

CPPFLAGS=$CPPFLAGS_BACKUP

###################################################################
########### GET LOW LEVEL DETAILS IN THE INSTALLATION #############
###################################################################

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

### Turn on runtime statistics
AC_ARG_ENABLE(stats,
    AS_HELP_STRING([--enable-stats],
    [turn on HC runtime stats (Default is false)]),
    [with_stats=$enableval],
    [with_stats=no;])

AS_IF([test "x$with_stats" != xno],
      [ AC_MSG_NOTICE([Enabled HC runtime statistics]) ],
      [ AC_MSG_NOTICE([Disabled HC runtime statistics]) ])

AM_CONDITIONAL(HC_STATS, test "x$with_stats" != xno)
### End runtime statistics

### Enable hwloc
AC_ARG_ENABLE(hwloc,
    AS_HELP_STRING([--enable-hwloc],
    [turn on hwloc (Default is false)]),
    [with_hwloc=$enableval],
    [with_hwloc=no;])

AS_IF([test "x$with_hwloc" != xno],
      [ AC_MSG_NOTICE([Enabled hwloc]) ],
      [ AC_MSG_NOTICE([Disabled hwloc]) ])

AM_CONDITIONAL(HC_HWLOC, test "x$with_hwloc" != xno)
### End runtime statistics

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

### SET DEPENDENCY COUNT
AC_ARG_WITH(max-dependency,
    AS_HELP_STRING([--with-max-dependency],
    [maximum number of dependencies for a task (Default is 4)]),
    [with_maxdependency=$withval
    AC_MSG_NOTICE([Maximum number of dependencies for a task is $with_maxdependency])
    AC_SUBST(MAX_DEPENDENCY, $with_maxdependency)
    AM_CONDITIONAL(HC_MAX_DEPENDENCY, true)],
    [AC_MSG_NOTICE([Maximum number of dependencies for a task uses default value])
    AM_CONDITIONAL(HC_MAX_DEPENDENCY, false)])
### End DEPENDENCY COUNT

### C++11 features check
_old_CXXFLAGS="${CXXFLAGS}"
CXXFLAGS="${_old_CXXFLAGS} -std=c++11"
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
