#!/bin/bash

set -e

# Bootstrap, Configure, Make and Install

#
# Defining some variables
#
PROJECT_NAME=hclib
INSTALL_PREFIX=${INSTALL_PREFIX:=${PWD}/${PROJECT_NAME}-install}
PREFIX_FLAGS="--prefix=$INSTALL_PREFIX"
: ${NPROC:=1}

# Don't clobber our custom header template
export AUTOHEADER="echo autoheader disabled"

#
# Bootstrap
#
# if install root has been specified, add --prefix option to configure
echo "[${PROJECT_NAME}] Bootstrap..."

./bootstrap.sh

for i in "$@"
do
case $i in
    --host=*)
    HOST="${i#*=}"
    echo "HOST is [${HOST}]"
    if [ "$HOST" = "honey64-unknown-hcos" ]; then
        cp tools/honeycomb/config.sub config/
    fi
    ;;
    *)
          # skip option
    ;;
esac
done

#
# Configure
#
echo "[${PROJECT_NAME}] Configure..."

REPO_ROOT=$PWD
COMPTREE=$PWD/compileTree
mkdir -p ${COMPTREE}

cd ${COMPTREE}

../configure ${PREFIX_FLAGS} $*

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


if [ "$HOST" != "honey64-unknown-hcos" ]; then
    echo "[${PROJECT_NAME}] Building system module..."
    cd ../modules/system
    HCLIB_ROOT=$INSTALL_PREFIX make install
fi

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
