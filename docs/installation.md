---
layout: home
title: Download and Installation
permalink: /installation/
---

HClib
=============================================

HClib is a task-based parallel programming model that supports the finish-async,
parallel-for, and future-promise parallel programming patterns through both C
and C++ APIs. HClib explicitly
exposes hardware locality of the hardware, while allowing the programmer to fall
back on sane defaults. The HClib runtime is a lightweight, work-stealing, and
locality-aware runtime. HClib is not itself an exascale programming system, but
is intended to be the intra-node resource management and scheduling component
within an exascale programming system, integrating with inter-node communication
models such as MPI, UPC++, or OpenSHMEM.

[![Build Status](https://travis-ci.org/habanero-rice/hclib.svg?branch=master)](https://travis-ci.org/habanero-rice/hclib)

Installation
---------------------------------------------

HClib follows your standard bootstrap, configure, and make installation
procedure. An install.sh script is provided for your convenience that will
build and install HClib. Simply run the script to install:

    ./install.sh

By default, HClib will be installed to `$PWD/hclib-install`. If you want to use
a different installation location, you can override the `INSTALL_PREFIX`
environment variable:

    INSTALL_PREFIX=/opt/local ./install.sh

Likewise, if you would like to use different C/C++ compilers other than the
system defaults, then you can specify them using the `CC` and `CXX` environment
variables. For example, if you want to use the Intel compilers:

    CC=icc CXX=icpc ./install.sh

You will need to set the `HCLIB_ROOT` environment variable to point to your
HClib installation directory. You can automatically set this variable after
installation by sourcing the `hclib_setup_env.sh` script. For example, assuming
HClib was installed with `INSTALL_PREFIX=/opt/local`:

    source /opt/local/bin/hclib_setup_env.sh


Dependencies
---------------------------------------------

* automake
* gcc >= 4.8.4, or clang >= 3.5
  (must support -std=c11 and -std=c++11)
* libxml2 (with development headers)


Testing
---------------------------------------------

The main regression tests for HClib are in the test/c and test/cpp folders. The
`test_all.sh` scripts in each of those folders will automatically build and run
all test cases.


Static Checks
---------------------------------------------

As part of the development workflow for HClib, any newly committed code should
be checked using standard static checking tools.

In particular, run cppcheck on all modified files. cppcheck is available online
at [1]. cppcheck should be run by cd-ing to tools/cppcheck and executing the
run.sh script from there (this assumes cppcheck is on your path). Any new errors
printed by cppcheck should be addressed before committing.

You should also run astyle on all modified files. astyle is a source code
auto-formatter. Simply cd to tools/astyle and execute the run.sh script from
there. This assumes you have astyle installed and it is on your path.

[1] https://sourceforge.net/projects/cppcheck/

