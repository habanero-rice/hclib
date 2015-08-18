=============================================
HABANERO-C++ LIBRARY INSTALLATION PROCEDURE
=============================================

1) Set the paths correctly in ./scripts/setup.sh

cd scripts

source ./setup.sh

cd ..

2) ./install.sh

=============================================
BUILDING TESTCASES
=============================================

1) Tests are inside "./test" 

2) Simply use "make" to build testcases

=============================================
EXECUTING TESTCASES
=============================================

1) Set total number of workers

export HCPP_WORKERS=<N>

2) See runtime statistics

export HCPP_STATS=1

3) Pin worker threads in round-robin fashion (supported only on Linux)

export HCPP_BIND_THREADS=1

4) Execute the testcase:

./a.out <command line args>
