#!/bin/bash

if ! [ -d ./scripts -a -f ./install.sh ]; then
    cat <<EOI
ERROR! This script should be run from the HClib project root directory.
Sample usage: ./scripts/clean.sh
EOI
    exit 1
fi

set -x # echo which files are being deleted

rm -f Makefile.in
rm -f src/Makefile.in
rm -f aclocal.m4
rm -rf autom4te.cache
rm -rf compileTree
rm -f configure
rm -rf hclib-install
