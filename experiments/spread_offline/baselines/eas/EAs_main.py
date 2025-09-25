import json
import os
from time import time
import torch
from baselines.eas.EAs_args import parse_args
from baselines.eas.EAs_experiments import (
    evaluation,
    sampling,
    train_proxies,
    report_stats,
)

        
def main(args):
    # log config
    if not os.path.exists(args.config_store_path):
        os.makedirs(args.config_store_path)
    name = args.task_name + "_" + args.mode + "_" + str(args.seed)
    config_dict = vars(args)
    with open(os.path.join(args.config_store_path, f"{name}.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_time = time()
    if args.mode == "train_proxies":
        train_proxies(args)
    elif args.mode == "sampling":
        sampling(args)
    elif args.mode == "evaluation":
        evaluation(args)
    elif args.mode == "report":
        report_stats(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"Total time: {time() - start_time} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
