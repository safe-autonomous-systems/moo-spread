#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p log_ms/spread/

problems=("dtlz4")
method="spread_ablation"
ablation="time"
timesteps=5000
seed=1000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem
    export method
    export timesteps
    export seed

    # Run training command
    python3 ms_main.py \
    --method="${method}" \
    --problem="${problem}" \
    --mode="train_spread" \
    --ablation="${ablation}" \
    --timesteps="${timesteps}" \
    --seed=0

    for nsample in 200 400 600 800; do
        export nsample
        python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --mode="sampling" \
        --nsample="${nsample}" \
        --timesteps="${timesteps}" \
        --seed="${seed}"
    done
done
