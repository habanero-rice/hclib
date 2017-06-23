#!/bin/bash

set -e

if [ $# -eq 1 ]; then
SIZE=$1
else
SIZE=large
fi

INPUT_FILE_1="./input/string1-$SIZE.txt"
INPUT_FILE_2="./input/string2-$SIZE.txt"

if [ "$SIZE" == "tiny" ]; then
        TILE_WIDTH=4
        TILE_HEIGHT=4
        INNER_TILE_WIDTH=2
        INNER_TILE_HEIGHT=2
        EXPECTED_RESULT=12
else
if [ "$SIZE" == "medium" ]; then
        TILE_WIDTH=232
        TILE_HEIGHT=240
        INNER_TILE_WIDTH=29
        INNER_TILE_HEIGHT=30
        EXPECTED_RESULT=3640
else
if [ "$SIZE" == "large" ]; then
        #TILE_WIDTH=2320
        #TILE_HEIGHT=2400
        TILE_WIDTH=232
        TILE_HEIGHT=240
        EXPECTED_RESULT=36472
else
if [ "$SIZE" == "huge" ]; then
        TILE_WIDTH=11600
        TILE_HEIGHT=12000
        INNER_TILE_WIDTH=725
        INNER_TILE_HEIGHT=750
        EXPECTED_RESULT=364792
fi
fi
fi
fi

export EXPECTED_RESULT
export PROJECT_RUN_ARGS="${INPUT_FILE_1} ${INPUT_FILE_2} ${TILE_WIDTH} ${TILE_HEIGHT}"

source ../common/run.sh
