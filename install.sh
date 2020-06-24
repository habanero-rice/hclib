#!/bin/bash

set -e

# Cmake, Make and Install

#
# Defining some variables
#
PROJECT_NAME=hclib
INSTALL_PREFIX=${INSTALL_PREFIX:=${PWD}/${PROJECT_NAME}-install}
: ${NPROC:=1}

#
# Cmake
#
echo "[${PROJECT_NAME}] Cmake..."

REPO_ROOT=$PWD
COMPTREE=$PWD/build
rm -rf ${COMPTREE}
mkdir -p ${COMPTREE}

cd ${COMPTREE}

cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} $* ..

#
# Make
#
echo "[${PROJECT_NAME}] Make..."
make VERBOSE=1 -j${NPROC}

#
# Make install
#
# if install root has been specified, perform make install
echo "[${PROJECT_NAME}] Make install... to ${INSTALL_PREFIX}"
make -j${NPROC} install


echo "[${PROJECT_NAME}] Building system module..."
cd ../modules/system
rm -rf build
mkdir build; cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} $* ..
make VERBOSE=1
make install

#
# Create environment setup script
#
# if install root has been specified, perform make install
HCLIB_ENV_SETUP_SCRIPT=${INSTALL_PREFIX}/bin/hclib_setup_env.sh

mkdir -p `dirname ${HCLIB_ENV_SETUP_SCRIPT}`
cat > "${HCLIB_ENV_SETUP_SCRIPT}" <<EOI
# HClib environment setup
export HCLIB_ROOT='${INSTALL_PREFIX}'

MY_OS=\$(uname -s)
if [ \${MY_OS} = "Darwin" ]; then
    export DYLD_LIBRARY_PATH=\${HCLIB_ROOT}/lib:\$DYLD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=\${HCLIB_ROOT}/lib:\$LD_LIBRARY_PATH
fi
EOI

cat <<EOI
[${PROJECT_NAME}] Installation complete.

${PROJECT_NAME} installed to: ${INSTALL_PREFIX}

Add the following to your .bashrc (or equivalent):
source ${HCLIB_ENV_SETUP_SCRIPT}
EOI
