#!/bin/bash

set -e

build_tests() {
    for test_dir in c cpp; do
        pushd ./test/$test_dir
        make clean
        make -j $1 all
        popd
    done
}

run_tests() {
    for test_dir in c cpp; do
        printf "\n>>>>>>>>> Testing %s strategy: %s <<<<<<<<<\n\n" "$1" "$test_dir"
        pushd ./test/$test_dir
        ./test_all.sh --skip-make
        popd
    done
}

if ! [ -d ./scripts -a -f ./install.sh ]; then
    cat <<EOI
ERROR! This script should be run from the HClib project root directory.
Sample usage: ./scripts/run-test-matrix.sh
EOI
    exit 1
elif [ -z "$HCLIB_ROOT" ]; then
    echo 'Missing $HCLIB_ROOT environment variable'
    exit 1
elif ! [ -d $HCLIB_ROOT ]; then
    echo "HClib installation not found at \$HCLIB_ROOT: $HCLIB_ROOT"
    exit 1
fi

# Test build settings in debug mode
build_tests
run_tests default

export HCLIB_WORKERS=1
run_tests 'default single thread'

config_matrix=(
  '--fixed'
  '--fixed --no-join'
  '--fixed --help-finish --help-global'
  '--threads'
  '--threads --no-join'
  '--threads --help-finish'
  '--fibers'
  '--fibers --no-join'
  '--fibers --help-finish'
)

export HCLIB_WORKERS=4
printf "\n>>>>>>>>> Switching to %d workers <<<<<<<<<\n\n" $HCLIB_WORKERS

for args in "${config_matrix[@]}"; do
    strategy=`echo "$args" | sed 's/--//g'`
    eval $($HCLIB_ROOT/bin/hclib-options $args) run_tests "'$strategy'"
done

# Test build settings in production mode
unset HCLIB_WORKERS
build_tests release
run_tests 'release default'

printf '\nTest matrix result: SUCCESS\n\n'

