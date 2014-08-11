#!/bin/bash

setenv CXX g++

# absolute path to the directories checked-out from github
setenv HCLIB_BASE   < ? >
setenv OCR_BASE  < ? >

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

cd ..
setenv BASE `pwd`
cd -

setenv HCLIB_ROOT ${HCLIB_BASE}/hclib-install
setenv OCR_ROOT ${OCR_BASE}/ocr-install
setenv HCPP_ROOT ${BASE}/hcpp-install
setenv OCR_CONFIG ${BASE}/machine-configs/mach-hcpp-1w.cfg
setenv LD_LIBRARY_PATH ${OCR_ROOT}/lib:${HCLIB_ROOT}/lib:${HCPP_ROOT}/lib:$LD_LIBRARY_PATH

