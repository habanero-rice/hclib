=============================================
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

=============================================
Installation
=============================================

HClib follows your standard bootstrap, configure, and make installation
procedure. An install.sh script is provided for your convenience while will
build and install HClib. To build HClib using this install script:

1) Set up the paths in scripts/setup.sh based on the provided template.

2) Source the setup script: cd scripts && source ./setup.sh && cd ..

3) ./install.sh
   a) If you would like to use different C/C++ compilers (the defaults are
      gcc/g++) then you can specify them using the CC and CXX environment
      variables at this step. For example, if I wanted to use the Intel
      compilers instead you would run:

          CC=icc CXX=icpc ./install.sh

=============================================
Dependencies
=============================================

gcc (>= 4.9.0, must support -std=c++11)

=============================================
Testing
=============================================

The main regression tests for HClib are in the test/c and test/cpp folders. The
test_all.sh scripts in each of those folders will automatically build and run
all test cases.


=============================================
Static Checks
=============================================

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
