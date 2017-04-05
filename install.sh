#!/bin/bash

set -e

# Bootstrap, Configure, Make and Install

#
# Defining some variables
#
PROJECT_NAME=hclib
PREFIX_FLAGS="--prefix=${INSTALL_PREFIX:=${PWD}/${PROJECT_NAME}-install}"
: ${NPROC:=1}

# Don't clobber our custom header template
export AUTOHEADER="echo autoheader disabled"

#
# Search for libtoolize
#
# if install root has been specified, add --prefix option to configure
echo "[${PROJECT_NAME}] Bootstrap..."


if type libtoolize &>/dev/null; then
    LIBTOOLIZE=`command -v libtoolize`
elif type glibtoolize &>/dev/null; then
    LIBTOOLIZE=`command -v glibtoolize`
else
    echo "ERROR: can't find libtoolize nor glibtoolize"
    exit 1
fi

aclocal -I config;

eval "$LIBTOOLIZE --force --copy"

autoreconf -vfi;

#
# Configure
#
echo "[${PROJECT_NAME}] Configure..."

COMPTREE=$PWD/compileTree
mkdir -p ${COMPTREE}

cd ${COMPTREE}

../configure ${PREFIX_FLAGS} ${HCUPC_FLAGS} ${HCLIB_FLAGS} ${HC_CUDA_FLAGS} $*

#
# Make
#
echo "[${PROJECT_NAME}] Make..."
make -j${NPROC}

#
# Make install
#
# if install root has been specified, perform make install
echo "[${PROJECT_NAME}] Make install... to ${INSTALL_PREFIX}"
make -j${NPROC} install

#
# Create environment setup script
#
# if install root has been specified, perform make install
HCLIB_ENV_SETUP_SCRIPT=${INSTALL_PREFIX}/bin/hclib_setup_env.sh

if [ -z `command -v xml2-config` ]; then
    echo "ERROR: Command xml2-config not found."\
         "Please ensure libxml2 (devel) is properly installed." >&2
    exit 1
fi

mkdir -p `dirname ${HCLIB_ENV_SETUP_SCRIPT}`
cat > "${HCLIB_ENV_SETUP_SCRIPT}" <<EOI
# HClib environment setup
export HCLIB_ROOT='${INSTALL_PREFIX}'
EOI

cat <<EOI
[${PROJECT_NAME}] Installation complete.

${PROJECT_NAME} installed to: ${INSTALL_PREFIX}

Add the following to your .bashrc (or equivalent):
source ${HCLIB_ENV_SETUP_SCRIPT}
EOI
