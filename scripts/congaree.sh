#!/bin/bash

export HCLIB_BASE=/home/kumar/openshmem/hcpp
export OPENSHMEM_ROOT=/home/kumar/openshmem/install-openshmem
export LIBXML2_INCLUDE=/usr/include/libxml2
export LIBXML2_LIBS=/usr/lib/x86_64-linux-gnu

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HABANERO-UPC
#
################################################

export SHMEM_CXX=oshcxx
export HCLIB_ROOT=${HCLIB_BASE}/hclib-install
export PATH=${OPENSHMEM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${HCLIB_ROOT}/lib:${LIBXML2_LIBS}:${OPENSHMEM_ROOT}/lib:$LD_LIBRARY_PATH
