#!/bin/bash

export CXX=g++

export HCLIB_ROOT=  < ? >
export OCR_ROOT=    < ? >
export OCR_INSTALL= < ? >

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

cd ..
BASE=`pwd`
cd -

export HCPP_INSTALL=${BASE}/hcpp-install
export OCR_CONFIG=${BASE}/machine-configs/mach-hcpp-1w.cfg
export LD_LIBRARY_PATH=${OCR_ROOT}/lib:${HCLIB_ROOT}/lib:${HCPP_INSTALL}/lib:$LD_LIBRARY_PATH

