#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z "$OMP_TO_HCLIB_HOME" ]]; then
    echo OMP_TO_HCLIB_HOME must be set to the root omp_to_hclib directory
    exit 1
fi

DRY_RUN=1

if [[ $# -eq 1 && $1 == doit ]]; then
    DRY_RUN=0
    echo "Updating files, not a dry run"
    sleep 5
else
    echo "Running dry run, run 'update_tests.sh doit' to actually update the test files"
fi
echo

for DIR in $(ls $SCRIPT_DIR); do
    if [[ -d $SCRIPT_DIR/$DIR && "$DIR" != "common" ]]; then
        NFOUND=$(find $OMP_TO_HCLIB_HOME/test -name "$DIR" | grep ref | wc -l)
        if [[ $NFOUND -eq 0 ]]; then
            echo "No test directories found for $DIR"
            exit 1
        elif [[ $NFOUND -ne 1 ]]; then
            echo "Found multiple test directories for $DIR ?"
            exit 1
        fi

        GENERATED_CODE_DIR=$(find $OMP_TO_HCLIB_HOME/test -name "$DIR" | grep ref)
        for C_FILE in $(find $GENERATED_CODE_DIR -name "*.c"); do
            FILENAME=$(basename $C_FILE)
            NACTUAL=$(find $DIR -name "$FILENAME" | wc -l)
            if [[ $NACTUAL -ne 1 ]]; then
                echo Found multiple matches for file $C_FILE
                find $DIR -name "$FILENAME"
                exit 1
            fi
            ACTUAL=$(find $DIR -name "$FILENAME")
            echo Updating $ACTUAL from $C_FILE
            if [[ $DRY_RUN -eq 0 ]]; then
                cp $C_FILE $ACTUAL
            fi
        done
        for CPP_FILE in $(find $GENERATED_CODE_DIR -name "*.cpp"); do
            FILENAME=$(basename $CPP_FILE)
            NACTUAL=$(find $DIR -name "$FILENAME" | wc -l)
            if [[ $NACTUAL -ne 1 ]]; then
                echo Found multiple matches for file $CPP_FILE
                find $DIR -name "$FILENAME"
                exit 1
            fi
            ACTUAL=$(find $DIR -name "$FILENAME")
            echo Updating $ACTUAL from $CPP_FILE
            if [[ $DRY_RUN -eq 0 ]]; then
                cp $CPP_FILE $ACTUAL
            fi
        done
    fi
done
