#!/bin/bash
export PYTHONPATH=$(pwd)

# Create log directory if it does not exist
mkdir -p logs/

probs=("zdt1" "zdt2" "zdt3" "dtlz2" "dtlz5" "dtlz7" "branincurrin" "penicillin" "carside")
switch=1

for prob in "${probs[@]}"; do
    echo "Starting processes for task: ${prob}"
    export prob

    if (( switch )); then 
        export switch_operator=1
        nohup python3 run_cdm_psl.py\
         --prob="${prob}"\
         --switch_operator=${switch} > logs/task_${prob}_cdm_psl_switch.log 2>&1
    else
        export switch_operator=0
        nohup python3 run_cdm_psl.py\
         --prob="${prob}"\
         --switch_operator=${switch} > logs/task_${prob}_cdm_psl.log 2>&1
    fi
done