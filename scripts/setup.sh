#!/bin/bash

#export TBB_MALLOC=/home/kumar/tbb
export LIBXML2_INCLUDE=/usr/local/Cellar/libxml2/2.9.2/include/libxml2
export LIBXML2_LIBS=/usr/local/Cellar/libxml2/2.9.2/lib
export BASE=/Users/vivek/rice_work/projects/release/hclib
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
