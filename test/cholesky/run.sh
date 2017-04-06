#!/bin/bash

make cholesky
rm -f cholesky.out

./cholesky 500 20 ./input/m_500.in

if ! cmp -s cholesky.out input/cholesky_out_500.txt; then
    echo "Test=Fail"
    exit 1
fi

echo "Test=Success"
