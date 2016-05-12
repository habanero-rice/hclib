#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z "$OMP_TO_HCLIB_HOME" ]]; then
    echo OMP_TO_HCLIB_HOME must be set to the root omp_to_hclib directory
    exit 1
fi

DRY_RUN=1
VERBOSE=0

for ARG in $*; do
    if [[ $ARG == doit ]]; then
        DRY_RUN=0
    elif [[ $ARG == verbose ]]; then
        VERBOSE=1
    else
        echo Unrecognized argument $ARG
        exit 1
    fi
done

if [[ $DRY_RUN -eq 0 ]]; then
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

        N_MATCHING_DIRECTORIES=$(find $OMP_TO_HCLIB_HOME/test -name "$DIR" | wc -l)
        if [[ $N_MATCHING_DIRECTORIES -ne 2 ]]; then
            echo Unexpected number of matching directories, expected 2 but got $N_MATCHING_DIRECTORIES
            exit 1
        fi

        GENERATED_CODE_DIR=$(find $OMP_TO_HCLIB_HOME/test -name "$DIR" | grep ref)
        REFERENCE_CODE_DIR=$(find $OMP_TO_HCLIB_HOME/test -name "$DIR" | grep -v ref)

        for C_FILE in $(find $GENERATED_CODE_DIR -name "*.c"); do
            FILENAME=$(basename $C_FILE)
            NACTUAL=$(find $DIR -name "$FILENAME" | wc -l)
            if [[ $NACTUAL -ne 1 ]]; then
                echo Found multiple matches for file $C_FILE
                find $DIR -name "$FILENAME"
                exit 1
            fi
            ACTUAL=$(find $DIR -name "$FILENAME")
            ANY_DELTA=$(diff $ACTUAL $C_FILE | wc -l)
            if [[ $ANY_DELTA -ne 0 ]]; then
                echo Updating $ACTUAL from $C_FILE
                if [[ $VERBOSE -eq 1 ]]; then
                    set +e
                    diff $ACTUAL $C_FILE
                    set -e
                    echo
                fi
                if [[ $DRY_RUN -eq 0 ]]; then
                    cp $C_FILE $ACTUAL
                fi
            fi

            NO_EXTENSION="${ACTUAL%.*}"
            cp $REFERENCE_CODE_DIR/$FILENAME $NO_EXTENSION.ref.c
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
            ANY_DELTA=$(diff $ACTUAL $CPP_FILE | wc -l)
            if [[ $ANY_DELTA -ne 0 ]]; then
                echo Updating $ACTUAL from $CPP_FILE
                if [[ $VERBOSE -eq 1 ]]; then
                    set +e
                    diff $ACTUAL $C_FILE
                    set -e
                    echo
                fi
                if [[ $DRY_RUN -eq 0 ]]; then
                    cp $CPP_FILE $ACTUAL
                fi
            fi

            NO_EXTENSION="${ACTUAL%.*}"
            cp $REFERENCE_CODE_DIR/$FILENAME $NO_EXTENSION.ref.cpp
        done
    fi
done
