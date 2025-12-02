#!/bin/bash
export PYTHONPATH=$(pwd)

problems=("dtlz4")
seed=1000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    export problem
    export seed

    for nsample in 200 400 600 800; do
        export nsample
        python3 run_moosvgd.py --problem="${problem}" \
        --mode="solve" --seed="${seed}" \
        --nsample="${nsample}"
    done

done