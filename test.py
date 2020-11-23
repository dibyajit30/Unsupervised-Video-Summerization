import numpy as np
import pandas as pd

import torch
from torch.utils import data

from utils import AverageMeter, seed_everything, get_filename
from PolicyGradient import REINFORCE
from dataset import VideoDataset

from tqdm.auto import tqdm

import argparse
import json

parser = argparse.ArgumentParser(
    description="Video summarization through Deep RL - test"
)
parser.add_argument(
    "--run_name", default="", type=str, required=True, help="name to identify exp"
)

args = parser.parse_args()
device = torch.device("cpu")


def load_dataloader(train_paths, val_paths):
    train_dataset = VideoDataset(train_paths)
    val_dataset = VideoDataset(val_paths)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_dataloader, val_dataloader


for fold in range(5):
    print("Fold::", fold)
    f = open(f"folds/fold_{fold}.json")
    dataset = json.load(f)
    f.close()
    train_paths = dataset["train"]
    val_paths = dataset["val"]
    train_dataloader, val_dataloader = load_dataloader(train_paths, val_paths)
    agent = REINFORCE(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        args=args,
        fold=fold,
        device=device,
    )
    agent.load_policy(f"models/{args.run_name}/model_{fold}.pth")
    agent.evaluate_policy(log=True)
