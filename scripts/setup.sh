#!/bin/bash

# Please set the following 4 environment variables

# Absolute path to hcpp directory
export HCLIB_BASE=/home/kumar/openshmem/hcpp

# Absolute path to installation directory of OpenSHMEM
export OPENSHMEM_ROOT=/home/kumar/openshmem/install-openshmem

# Path to libxml2 headers
export LIBXML2_INCLUDE=/usr/include/libxml2

# Path to libxml2 libraries
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
