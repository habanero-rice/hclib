#!/bin/bash

set -e

rm -rf compileTree hclib-install

BASE_PREFIX=$PWD/hclib-install/multi

env INSTALL_PREFIX=$PWD/hclib-install ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/fibers $(./scripts/hclib-options --fibers) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/non-blocking $(./scripts/hclib-options --fixed --help-global) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/threads $(./scripts/hclib-options --threads) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/threads-help $(./scripts/hclib-options --threads --help-finish) ./install.sh
rm -rf compileTree

# default is fibers + help-finish
cd $BASE_PREFIX
ln -s .. fibers-help
ln -s .. default
