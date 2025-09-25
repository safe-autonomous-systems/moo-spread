#!/bin/bash
export PYTHONPATH=$(pwd)

mkdir -p log_ms/spread/

tasks=("zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "dtlz1"  "dtlz2" "dtlz3" "dtlz4" "dtlz5" "dtlz6" "dtlz7"
 "re21" "re22" "re25" "re31" "re32" "re33" "re35" "re36" "re37" "re41" "re42" "re61") 
method="spread"

for task in "${tasks[@]}"; do
    echo "Starting processes for task: ${task}"
    export task

    # Run training commands
    nohup python3 off_ms_main.py --task_name="${task}" --mode="train_proxies"\
     --seed=0 > log_ms/spread/task_${task}_proxies_training.log 2>&1
    nohup python3 off_ms_main.py --task_name="${task}" --mode="train_spread"\
     --seed=0 > log_ms/spread/task_${task}_spread_training.log 2>&1

    # Run sampling commands for different seeds
    for seed in 1000 2000 3000 4000 5000; do
        export seed
        nohup python3 off_ms_main.py --task_name="${task}" --mode="sampling"\
         --seed="${seed}" > log_ms/spread/sampling_task_${task}_seed_${seed}.log 2>&1
        python3 off_ms_main.py --task_name="${task}" --mode="evaluation"\
         --seed="${seed}"
    done

    # Finally, run the report command (in the foreground)
    python3 off_ms_main.py --task_name="${task}" --mode="report"
done