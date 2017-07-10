#!/bin/bash

set -e

rm -rf compileTree hclib-install

BASE_PREFIX=$PWD/hclib-install/multi

env INSTALL_PREFIX=$PWD/hclib-install $(./scripts/hclib-options --no-join --fibers --help-finish) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/fibers $(./scripts/hclib-options --no-join --fibers) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/non-blocking $(./scripts/hclib-options --no-join --fixed --help-global) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/threads $(./scripts/hclib-options --no-join --threads) ./install.sh
rm -rf compileTree

env INSTALL_PREFIX=$BASE_PREFIX/threads-help $(./scripts/hclib-options --no-join --threads --help-finish) ./install.sh
rm -rf compileTree

# default is fibers + help-finish
cd $BASE_PREFIX
ln -s .. fibers-help
ln -s .. default
