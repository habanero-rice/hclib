#!/bin/bash

set -e

# On DAVINCI, reconfigure.sh
# On Edison, reconfigure.sh -lxpmem -lugni
# On Titan, reconfigure.sh -lxpmem -lugni
EXTRA_LIBS=$*

INDEX=1
for OPTION in $(ls $GASNET_INSTALL/lib/libgasnet-*.a); do
    echo $INDEX $OPTION
    INDEX=$(($INDEX+1))
done

GASNET_LIB=
if [[ $INDEX -eq 2 ]]; then
    # Only one gasnet library found, use it
    echo Only one GASNET library, using it.
    GASNET_LIB=$(ls $GASNET_INSTALL/lib/libgasnet-*.a)
else
    echo Select a GASNET library to use:
    read OPTION_INDEX

    INDEX=1
    for OPTION in $(ls $GASNET_INSTALL/lib/libgasnet-*.a); do
        if [[ $OPTION_INDEX -eq $INDEX ]]; then
            GASNET_LIB=$OPTION
        fi
        INDEX=$(($INDEX+1))
    done
fi

if [[ -z "$GASNET_LIB" ]]; then
    echo Invalid selection
    exit 1
fi

CONDUIT_NAME=$(echo $(basename $GASNET_LIB) | cut -d '-' -f2)
CONDUIT_TYPE=$(echo $(basename $GASNET_LIB) | cut -d '-' -f3 | cut -d '.' -f1)

sed -e "s|CONDUIT_NAME|$CONDUIT_NAME|g" \
        inc/hclib_openshmem.post.mak.template > inc/hclib_openshmem.post.mak
sed -i -e "s|CONDUIT_TYPE|$CONDUIT_TYPE|g" inc/hclib_openshmem.post.mak
sed -i -e "s|EXTRA_LIBS_PATTERN|$EXTRA_LIBS|g" inc/hclib_openshmem.post.mak
make clean
make
