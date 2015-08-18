#!/bin/bash

#export TBB_MALLOC=/Users/vivek/rice_work/projects/tbb/mac
export HCPP_FLAGS="--enable-asserts --enable-commStats"
export LIBXML2=/usr/local/Cellar/libxml2/2.9.2

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

cd ..
BASE=`pwd`
cd -
export hcpp=${BASE}
export HCPP_ROOT=${hcpp}/hcpp-install
export LD_LIBRARY_PATH=${HCPP_ROOT}/lib:$LD_LIBRARY_PATH

if [ ! -z "${TBB_MALLOC}" ]; then
   export LD_LIBRARY_PATH=${TBB_MALLOC}:$LD_LIBRARY_PATH
fi
