#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p log_ms/paretoflow/

tasks=("zdt1" "zdt2" "zdt3" "zdt4" "zdt6" "dtlz1"  "dtlz2" "dtlz3" "dtlz4" "dtlz5" "dtlz6" "dtlz7"
 "re21" "re22" "re25" "re31" "re32" "re33" "re35" "re36" "re37" "re41" "re42" "re61") 

for task in "${tasks[@]}"; do
    echo "Starting processes for task: ${task}"
    # You can set the task variable if other scripts need it exported
    export task

    # Run training commands
    python3 -m baselines.paretoflow.paretoflow_main\
     --task_name="${task}" --mode="train_proxies" --seed=0 
    python3 -m baselines.paretoflow.paretoflow_main\
     --task_name="${task}" --mode="train_flow_matching" --seed=0 

    # Run sampling commands for different seeds
    for seed in 1000 2000 3000 4000 5000; do
        export seed
        python3 -m baselines.paretoflow.paretoflow_main\
         --task_name="${task}" --mode="sampling"\
          --seed="${seed}"
    done

    # Finally, run the report command (in the foreground)
    python3 -m baselines.paretoflow.paretoflow_main\
     --task_name="${task}" --mode="report"
done