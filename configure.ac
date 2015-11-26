# check required autoconf version
AC_PREREQ([2.59])

# define distribution properties
AC_INIT([hcpp], [0.1], [vivekk@rice.edu])

# check if srcdir is correct
AC_CONFIG_SRCDIR([inc/hcpp.h])

AC_CONFIG_AUX_DIR([config])

AM_INIT_AUTOMAKE([1.9 tar-ustar no-define foreign dist-bzip2])

AC_COPYRIGHT([])

# configure script revision
AC_REVISION([$Revision: 0.1 $])

AC_PROG_CXX
AC_PROG_LIBTOOL
AC_LANG(C++)

CPPFLAGS=$CPPFLAGS_BACKUP

###################################################################
########### GET LOW LEVEL DETAILS IN THE INSTALLATION #############
###################################################################

# Conditional flag to build for the hcpp/hcupc++ library

### INTERFACE HCPP WITH PHASER LIBRARY
AC_ARG_ENABLE(phaser,
    AS_HELP_STRING([--enable-phaser],
    [enable interface with Phaser (Default is false)]),
    [with_phaser=$enableval],
    [with_phaser=no;])

AS_IF([test "x$with_phaser" != xno],
      [ AC_MSG_NOTICE([Enabled Phaser support]) ],
      [ AC_MSG_NOTICE([Disabled Phaser support]) ])

AM_CONDITIONAL(_PHASERS_, test "x$with_phaser" != xno)
### End OCR

### INTERFACE HCPP WITH OCR
AC_ARG_ENABLE(ocr,
    AS_HELP_STRING([--enable-ocr],
    [enable interface with OCR (Default is false)]),
    [with_ocr=$enableval],
    [with_ocr=no;])

AS_IF([test "x$with_ocr" != xno],
      [ AC_MSG_NOTICE([Enabled OCR support]) ],
      [ AC_MSG_NOTICE([Disabled OCR support]) ])

AM_CONDITIONAL(WITH_OCR, test "x$with_ocr" != xno)
### End OCR

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
