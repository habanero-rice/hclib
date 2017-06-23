#!/bin/bash

set -e

rm -rf compileTree hclib-install

BASE_PREFIX=$PWD/hclib-install/multi

env INSTALL_PREFIX=$BASE_PREFIX/default ./install.sh

rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/non-blocking $(./scripts/hclib-options --fixed --help-global) ./install.sh

rm -rf compileTree

cd hclib-install
for dir in include bin; do
    ln -s multi/default/$dir .
done
