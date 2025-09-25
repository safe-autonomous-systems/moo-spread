#!/bin/bash
export PYTHONPATH=$(pwd)

problems=("dtlz4")
method="hvgrad"
seed=1000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    export problem
    export seed

    for nsample in 200 400 600 800; do
        export nsample
        python3 run_hvgrad.py --method="${method}" \
        --problem="${problem}" --mode="solve" \
        --seed="${seed}" --nsample="${nsample}"
    done
done