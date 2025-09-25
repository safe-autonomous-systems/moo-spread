#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p log_ms/spread/

problems=("zdt2" "dtlz4" "re41")
method="spread_ablation"
ablation="ditblocks"
timesteps=5000
seed=1000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem
    export method
    export timesteps
    export seed
    export ablation

    for num_blocks in 1 2 3 4 5; do
        export num_blocks

        # Run training command
        python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --ablation="${ablation}" \
        --num_blocks="${num_blocks}" \
        --mode="train_spread" \
        --timesteps="${timesteps}" \
        --seed=0 

        python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --ablation="${ablation}" \
        --mode="sampling" \
        --num_blocks="${num_blocks}" \
        --timesteps="${timesteps}" \
        --seed="${seed}" 
    done
done
