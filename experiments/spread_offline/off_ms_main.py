import json
import os
from time import time
import torch
from off_ms_args import parse_args
from off_ms_experiments import (
    evaluation,
    sampling,
    train_proxies,
    train_spread,
    report_stats,
)

import warnings
warnings.filterwarnings("ignore")


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
    elif args.mode == "train_spread":
        train_spread(args)
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
