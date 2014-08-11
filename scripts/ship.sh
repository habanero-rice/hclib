#!/bin/bash

cd $hcpp
./clobber.sh

cd $hcpp/test
for i in `find . -name Makefile`
do
  cd `dirname $i`
  make clean
  cd -
done

cd $hcpp

rm -rf hcpp-install
