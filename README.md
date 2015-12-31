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
