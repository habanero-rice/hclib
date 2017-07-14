#!/bin/bash

echo Using ${HCLIB_WORKERS:=8} workers

export MY_PREFIXES="gh gh heap_args ss4k escaping"
./run.sh
