# check required autoconf version
AC_PREREQ([2.59])

# define distribution properties
AC_INIT([hcpp], [0.1], [hc-dev-l@mailman.rice.edu])

# check if srcdir is correct
AC_CONFIG_SRCDIR([inc/hcpp.h])

AC_CONFIG_AUX_DIR([config])

AM_INIT_AUTOMAKE([1.9 tar-ustar no-define foreign dist-bzip2])

AC_COPYRIGHT([todo_copyright_info])

# configure script revision
AC_REVISION([$Revision: 0.1 $])

AC_PROG_CXX
AC_PROG_LIBTOOL
AC_LANG(C++)

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

# Generates and runs config.status, which in turn creates the makefiles 
# and any other files resulting from configuration
AC_OUTPUT

