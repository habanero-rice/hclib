#!/bin/bash -l

set -e

ulimit -c unlimited
export HCLIB_WORKERS=64
export OMP_NUM_THREADS=$HCLIB_WORKERS
export TBB_NUM_THREADS=$HCLIB_WORKERS
export MODELS="hclib iomp tbb"

# 2 sockets x 12-core CPUs

cd $HCLIB_HOME/test/performance-regression/micro

rm -f *_iomp

make clean
CXX=icpc CC=icc make -j hclib omp tbb
for F in $(ls *_omp); do mv $F ${F:0:$((${#F} - 4))}_iomp; done

LOG_FILE=metrics.csv
rm -f $LOG_FILE

LINE="metric,dataset"
for MODEL in $MODELS; do
    LINE="$LINE,$MODEL"
done
LINE="$LINE,,"
echo $LINE >> $LOG_FILE

for TEST in task_spawn future_spawn task_wait_flat task_wait_recursive fan_out \
        fan_out_and_in bin_fan_out parallel_loop prod_cons reduce_var \
        unbalanced_bin_fan_out; do
    echo
    echo "========== Testing ${TEST} =========="
    echo

    for MODEL in $MODELS; do
        EXE=${TEST}_${MODEL}
        ARGS=
        if [[ $MODEL == 'realm' ]]; then
            ARGS="-ll:cpu 64"
        fi

        if [[ -f $EXE ]]; then
            echo "========== $EXE =========="
            ./$EXE $ARGS &> $MODEL
        fi
    done

    UNIQUE_METRICS=$(cat hclib | grep "^METRIC " | awk '{ print $2 }' | sort | \
        uniq)
    for METRIC in $UNIQUE_METRICS; do
        DATASET=$(cat hclib | grep "^METRIC $METRIC " | head -n 1 | \
            awk '{ print $3 }')

        LINE="$METRIC,$DATASET"
        for MODEL in $MODELS; do
            if [[ -f $MODEL ]]; then
                PERF=$(cat $MODEL | grep "^METRIC $METRIC" | \
                    awk '{ print $4 }' | sort -n -r | head -n 1)
                LINE="$LINE,$PERF"
            else
                LINE="$LINE,"
            fi
        done
        LINE="$LINE,,"
        echo $LINE >> $LOG_FILE
    done

    # Cleanup log files
    for MODEL in $MODELS; do
        EXE=${TEST}_${MODEL}
        if [[ -f $EXE ]]; then
            rm $MODEL
        fi
    done
done
