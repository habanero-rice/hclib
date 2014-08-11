#! /bin/csh -f

setenv CXX g++

setenv HCLIB_ROOT   < ? >
setenv OCR_ROOT     < ? >
setenv OCR_INSTALL  < ? >

################################################
#
# DO NOT MODIFY ANYTHING BELOW UNLESS YOU ARE
# CHANGING THE INSTALLATION PATH OF HCPP
#
################################################

cd ..
setenv BASE `pwd`
cd -

setenv HCPP_INSTALL ${BASE}/hcpp-install
setenv OCR_CONFIG ${BASE}/machine-configs/mach-hcpp-1w.cfg
setenv LD_LIBRARY_PATH ${OCR_ROOT}/lib:${HCLIB_ROOT}/lib:${HCPP_INSTALL}/lib:$LD_LIBRARY_PATH
