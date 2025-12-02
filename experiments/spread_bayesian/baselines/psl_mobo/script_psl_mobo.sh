#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p logs/

probs=("zdt1" "zdt2" "zdt3" "dtlz2" "dtlz5" "dtlz7" "branincurrin" "penicillin" "carside")

for prob in "${probs[@]}"; do
    echo "Starting processes for task: ${prob}"
    # You can set the task variable if other scripts need it exported
    export prob

    nohup python3 run_psl_mobo.py\
     --prob="${prob}" > logs/task_${prob}_psl_mobo.log 2>&1
done
