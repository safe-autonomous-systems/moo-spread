#!/bin/bash
export PYTHONPATH=$(pwd)
# Use command: chmod u+x script_spread_online.sh
# to make this script executable.

# Create log directory if it does not exist
mkdir -p log_ms/spread/

problems=("zdt1" "zdt2" "zdt3" "re21" "dtlz2" "dtlz4" "dtlz7" "re33" "re34" "re37" "re41")

method="spread"
timesteps=5000


for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem
    export method
    export timesteps

    # Run training command
    python3 ms_main.py \
    --method="${method}" \
    --problem="${problem}" \
    --mode="train_spread" \
    --timesteps="${timesteps}" \
    --seed=0 

    # Run sampling commands for different seeds
    for seed in 1000 2000 3000 4000 5000; do
        export seed
        python3 ms_main.py \
        --method="${method}" \
        --problem="${problem}" \
        --mode="sampling" \
        --timesteps="${timesteps}" \
        --seed="${seed}" 
    done

    # Finally, run the report command (in the foreground)
    python3 ms_main.py \
    --method="${method}" \
    --problem="${problem}" \
    --timesteps="${timesteps}" \
    --mode="report"
done
