#!/bin/sh

set -e

#
# Defining some variables
#

PROJECT_NAME=hcpp

check_error()
{
    if [ $# -gt 2 ]; then
        echo "Error in check_error call";
        exit 1;
    fi;
    ERRCODE="$1";
    if [ "$ERRCODE" = "0" ]; then
        return 0;
    fi;
    if [ $# -eq 2 ]; then
        ERRMESSAGE="$2";
    else
        ERRMESSAGE="Error";
    fi;
    echo "[${PROJECT_NAME}] $ERRMESSAGE";
    exit $ERRCODE;
}


#
# Bootstrap, Configure, Make and Install
#

if [ -z "$NPROC" ]; then 
    NPROC=1
fi

#
# Bootstrap
#
# if install root has been specified, add --prefix option to configure
if [ -n "${INSTALL_ROOT}" ]; then
    INSTALL_ROOT="--prefix=${INSTALL_ROOT}"
else
    INSTALL_ROOT="--prefix=${PWD}/${PROJECT_NAME}-install"
fi

echo "[${PROJECT_NAME}] Bootstrap..."

./bootstrap.sh
check_error "$?" "Bootstrap failed";

#
# Configure
#
echo "[${PROJECT_NAME}]] Configure..."

COMPTREE=$PWD/compileTree
mkdir -p ${COMPTREE}

cd ${COMPTREE}

../configure ${INSTALL_ROOT} ${HCUPC_FLAGS} ${HCPP_FLAGS}
check_error "$?" "Configure failed";

#
# Make
#
echo "[${PROJECT_NAME}]] Make..."
make -j${NPROC}
check_error "$?" "Build failed";

#
# Make install
#
# if install root has been specified, perform make install
echo "[${PROJECT_NAME}]] Make install... to ${INSTALL_ROOT}"
make -j${NPROC} install
check_error "$?" "Installation failed";


echo "[${PROJECT_NAME}]] Installation complete."
