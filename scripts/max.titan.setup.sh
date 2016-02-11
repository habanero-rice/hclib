#!/bin/bash

# export TBB_MALLOC=/home/kumar/tbb
# export HCLIB_FLAGS="--enable-production"
export LIBXML2_INCLUDE=/usr/include/libxml2
export LIBXML2_LIBS=/usr/lib64
export BASE=/ccs/home/jmg3/hcpp
export HC_CUDA_FLAGS=--enable-cuda

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCLIB
#
################################################

export hclib=${BASE}
export HCLIB_ROOT=${hclib}/hclib-install
export LD_LIBRARY_PATH=${HCLIB_ROOT}/lib:${LIBXML2_LIBS}:$LD_LIBRARY_PATH

if [ ! -z "${TBB_MALLOC}" ]; then
   export LD_LIBRARY_PATH=${TBB_MALLOC}:$LD_LIBRARY_PATH
fi
