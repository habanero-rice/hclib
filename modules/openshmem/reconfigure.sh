#!/bin/bash

set -e

declare -A GASNET_OPTIONS

INDEX=1
for OPTION in $(ls $GASNET_INSTALL/lib/libgasnet-*.a); do
    GASNET_OPTIONS[$INDEX]=$OPTION
    INDEX=$(($INDEX + 1))
done

GASNET_LIB=
if [[ $INDEX -eq 2 ]]; then
    # Only one gasnet library found, use it
    GASNET_LIB=${GASNET_OPTIONS[1]}
else
    for INDEX in ${!GASNET_OPTIONS[@]}; do
        echo $INDEX ${GASNET_OPTIONS[$INDEX]}
    done
    echo Select a GASNET library to use:
    read OPTION_INDEX
    GASNET_LIB=${GASNET_OPTIONS[$OPTION_INDEX]}
fi

if [[ -z "$GASNET_LIB" ]]; then
    echo Invalid selection
    exit 1
fi

sed -e "s|GASNET_ROOT_PATTERN|$GASNET_INSTALL|g" \
        inc/hclib_openshmem.post.mak.template > inc/hclib_openshmem.post.mak
sed -i -e "s|GASNET_LIB_PATTERN|$GASNET_LIB|g" inc/hclib_openshmem.post.mak
make clean
make
