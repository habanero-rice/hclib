#!/bin/bash

for n in 9 18 36 72; do
time env HCLIB_WORKERS=$n ./run-all.sh
done
