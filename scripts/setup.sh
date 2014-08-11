unset HCPP_INSTALL
unset HCLIB_ROOT
unset OCR_CONFIG
unset OCR_INSTALL
unset OCR_ROOT
unset hcpp
unset CXX
unset build

cd ../..
BASE=`pwd`
cd -

export CXX=g++

export hcpp=${BASE}/hcpp
export build=${hcpp}/scripts/build.sh

export HCPP_INSTALL=${hcpp}/hcpp-install
export HCLIB_ROOT=${BASE}/hclib/hclib-install
export OCR_ROOT=${BASE}/ocr/ocr-install
export OCR_INSTALL=${BASE}/ocr/ocr-install
export OCR_CONFIG=${hcpp}/machine-configs/mach-hcpp-2w.cfg

export LD_LIBRARY_PATH=${OCR_ROOT}/lib:${HCLIB_ROOT}/lib:${HCPP_INSTALL}/lib:${LD_LIBRARY_PATH}
