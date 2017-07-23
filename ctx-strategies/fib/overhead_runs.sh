#!/bin/bash

echo Using ${HCLIB_WORKERS:=8} workers

export MY_PREFIXES="gh gh nb heap_args ss4k escaping cxx"
./run.sh
