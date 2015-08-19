
Habanero-C++ library supports two different runtime backends: a) Light Weight Standalone Runtime (LWSR) and Open Community Runtime (OCR). Please find the instructions related to both the backends below.

=============================================
HABANERO-C++ LIBRARY INSTALLATION PROCEDURE
=============================================

----------> LWSR backend

1) Set the paths correctly in ./scripts/setup.sh

cd scripts

source ./setup.sh

cd ..

2) ./install.sh

----------> OCR backend

1) Clone hclib:

git clone https://github.com/habanero-rice/hclib.git

2) Follow the instructions inside hclib to build hclib with OCR support. Set the environment variables mentioned inside hclib properly.

3) Once hclib and OCR is installed, do this in hcpp directory:

a) Set the paths correctly in ./scripts/setup.sh

cd scripts

source ./setup.sh

cd ..

b) HCPP_FLAGS="--enable-ocr" ./install.sh

=============================================
BUILDING TESTCASES
=============================================

For both LWSR and OCR backend, setup all environment variables properly (see above)

1) Tests are inside "./test" 

2) Simply use "make" to build testcases

=============================================
EXECUTING TESTCASES
=============================================

For both LWSR and OCR backend, setup all environment variables properly (see above)

----------> LWSR backend

1) Set total number of workers

a) export HCPP_WORKERS=N

OR 

b) use an HPT xml file. Some sample files in directory ./hpt

export HCPP_HPT_FILE=/absolute/path/hcpp/hpt/hpt-testing.xml

2) See runtime statistics

export HCPP_STATS=1

3) Pin worker threads in round-robin fashion (supported only on Linux)

export HCPP_BIND_THREADS=1

4) Execute the testcase:

./a.out command_line_args

----------> OCR backend

1) The work-stealing configuration file, which declares the total number of worker threads (OCR_CONFIG) is required to configure the total number of runtime workers (details in hclib repository). Then source the setup.sh script in hcpp repository (as mentioned earlier in this file).

2) Execute the testcase:

./a.out command_line_arg

