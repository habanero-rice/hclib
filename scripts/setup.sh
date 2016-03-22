#!/bin/bash

export TBB_MALLOC=/home/kumar/tbb
#export HCPP_FLAGS="--enable-production"
export LIBXML2_INCLUDE=/usr/include/libxml2
export LIBXML2_LIBS=/usr/lib/x86_64-linux-gnu
export BASE=/home/kumar/hcpp
################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

export hcpp=${BASE}
export HCPP_ROOT=${hcpp}/hcpp-install
export LD_LIBRARY_PATH=${HCPP_ROOT}/lib:${LIBXML2_LIBS}:$LD_LIBRARY_PATH

if [ ! -z "${TBB_MALLOC}" ]; then
   export LD_LIBRARY_PATH=${TBB_MALLOC}:$LD_LIBRARY_PATH
fi
