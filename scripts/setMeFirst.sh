#!/bin/bash

export CXX=g++

# absolute path to the directories checked-out from github
export HCLIB_BASE=  < ? >
export OCR_BASE= < ? >

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

cd ..
BASE=`pwd`
cd -

export HCLIB_ROOT=${HCLIB_BASE}/hclib-install
export OCR_ROOT=${OCR_BASE}/ocr-install
export HCPP_ROOT=${BASE}/hcpp-install
export OCR_CONFIG=${BASE}/machine-configs/mach-hcpp-1w.cfg
export LD_LIBRARY_PATH=${OCR_ROOT}/lib:${HCLIB_ROOT}/lib:${HCPP_ROOT}/lib:$LD_LIBRARY_PATH

