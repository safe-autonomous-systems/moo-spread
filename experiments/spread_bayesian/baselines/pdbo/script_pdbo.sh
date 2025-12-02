#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p logs/

probs=("zdt1" "zdt2" "zdt3" "dtlz2" "dtlz5" "dtlz7" "branincurrin" "penicillin" "carside")

for prob in "${probs[@]}"; do
    echo "Starting processes for task: ${prob}"

    export prob
    nohup python3 run_pdbo.py\
     --problem="${prob}" > logs/pdbo_task_${prob}_${start_run}.log 2>&1
done
