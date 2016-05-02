#!/bin/bash

# export TBB_MALLOC=/home/kumar/tbb
# export HCLIB_FLAGS="--enable-production"
export LIBXML2_INCLUDE=/usr/include/libxml2
export LIBXML2_LIBS=/usr/lib64
export BASE=$PROJ_DIR/hclib

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
