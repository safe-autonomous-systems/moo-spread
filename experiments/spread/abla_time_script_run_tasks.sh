#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p log_ms/spread/

problems=("zdt1" "dtlz2" "re41") 

method=spread_ablation
ablation=time
timesteps=5000

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem
    export method
    export ablation
    export timesteps

    # Run training command
    nohup python3 ms_main.py --method="${method}" \
    --ablation="${ablation}" \
     --problem="${problem}" \
     --timesteps="${timesteps}" \
     --mode="train_spread" \
     --seed=0 

    for seed in 1000; do
        export seed
        nohup python3 ms_main.py \
        --method="${method}" \
        --ablation="${ablation}" \
        --problem="${problem}" \
        --timesteps="${timesteps}" \
        --mode="sampling" \
        --seed="${seed}"
    done
done