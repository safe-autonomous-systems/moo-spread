#!/bin/bash
export PYTHONPATH=$(pwd)
# Create log directory if it does not exist
mkdir -p log_ms/spread/

problems=("zdt2" "dtlz4" "re41") 
method="spread_ablation"
ablation="lambda_rep"
timesteps=5000
seed=1000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem
    export method
    export timesteps
    export ablation
    export seed

    python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --mode="train_spread" \
        --timesteps="${timesteps}" \
        --ablation="${ablation}" \
        --seed=0

    python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --mode="sampling" \
        --timesteps="${timesteps}" \
        --ablation="${ablation}" \
        --seed="${seed}"
    done
done
