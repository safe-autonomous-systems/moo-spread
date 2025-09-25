#!/bin/bash
export PYTHONPATH=$(pwd)
# Use command: chmod u+x script_hvgrad.sh
# to make this script executable.

problems=("zdt1" "zdt2" "zdt3" "re21" "dtlz2" "dtlz4" "dtlz7" "re33" "re34" "re37" "re41")

method="hvgrad"

for problem in "${problems[@]}"; do
    echo "Starting processes for problem: ${problem}"
    # You can set the problem variable if other scripts need it exported
    export problem

    # Run sampling commands for different seeds
    for seed in 1000 2000 3000 4000 5000; do
        export seed
        python3 run_hvgrad.py --method="${method}" --problem="${problem}" --mode="solve" --seed="${seed}"
    done

    # Finally, run the report command (in the foreground)
    python3 run_hvgrad.py --method="${method}" --problem="${problem}" --mode="report"
done