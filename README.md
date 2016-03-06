=============================================
HABANERO-C++ LIBRARY INSTALLATION PROCEDURE
=============================================

1) Set the paths correctly in ./scripts/setup.sh

cd scripts && source ./setup.sh && cd ..

2) If you are compiling HClib with CUDA support, ensure the CUDA_HOME
   environment variable is set.

3) If you want to generate your own HPT configuration, you will need HWLOC
   installed and the environment variable HWLOC_HOME pointed to its root
   installation directory with $HWLOC_HOME/lib on your LD_LIBRARY_PATH.

4) ./install.sh
   a) If you would like to use different C/C++ compilers (the defaults are
      gcc/g++) then you can specify them using the CC and CXX environment
      variables at this step. For example, if I wanted to use icc instead I
      would run:

          CC=icc CXX=icpc ./install.sh

=============================================
DEPENDENCIES
=============================================

gcc (>= 4.9.0, must support -std=c++11)

=============================================
BUILDING TESTCASES
=============================================

Setup all environment variables properly (see above)

1) Tests are inside "./test" 

2) Simply use "make" to build testcases

=============================================
EXECUTING TESTCASES
=============================================

Setup all environment variables properly (see above)

1) Set total number of workers

a) export HCLIB_WORKERS=N

OR 

b) use an HPT xml file. Some sample files in directory ./hpt

export HCLIB_HPT_FILE=/absolute/path/hclib/hpt/hpt-testing.xml

2) See runtime statistics

export HCLIB_STATS=1

3) Pin worker threads in round-robin fashion (supported only on Linux)

export HCLIB_BIND_THREADS=1

4) Execute the testcase:

./a.out command_line_args

=============================================
STATIC CHECKS
=============================================

As part of the development workflow for HClib, any newly committed code should
be checked using standard static checking tools.

In particular, run cppcheck on all modified files. cppcheck is available online
at [1]. cppcheck should be run by cd-ing to tools/cppcheck and executing the
run.sh script from there (this assumes cppcheck is on your path). Any new errors
printed by cppcheck should be addressed before committing.

You should also run astyle on all modified fields. astyle is a source code
auto-formatter. Simply cd to tools/astyle and execute the run.sh script from
there. This assumes you have astyle installed and it is on your path.

[1] https://sourceforge.net/projects/cppcheck/
