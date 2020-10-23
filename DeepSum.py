import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.utils import data
import json
import logging
from utils import AverageMeter, seed_everything, get_filename
from reward import compute_reward
import argparse
from tqdm.auto import tqdm
import glob

parser = argparse.ArgumentParser(description="Video summarization through Deep RL")
parser.add_argument(
    "--run_name", default="", type=str, required=True, help="name to identify exp",
)
parser.add_argument(
    "--epochs", default=60, type=int, help="number of epochs",
)
parser.add_argument(
    "--num_episodes", default=5, type=int, help="number of episodes",
)
parser.add_argument(
    "--seed", default=1, type=int, required=True, help="seed",
)
args = parser.parse_args()

seed_everything(seed=args.seed)
logging.basicConfig(
    filename=f"logs/{args.run_name}.log", level=logging.INFO, format="%(message)s",
)


class VideoDataset(data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def __getitem__(self, i):
        path = self.paths[i]
        id = get_filename(path)
        feature = torch.load(path)
        return feature, id

    def __len__(self):
        return len(self.paths)


class PolicyNet(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        out = torch.sigmoid(self.head(h))
        return out


class REINFORCE:
    def __init__(
        self,
        train_paths,
        val_paths,
        args,
        fold,
        gamma=0.99,
        beta=0.01,
        lr=0.001,
        device="cpu",
    ):
        self.policy = PolicyNet()
        self.policy.to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        self.baselines = self.load_baselines(train_paths)

        self.train_dataloader, self.val_dataloader = self.load_dataloader(
            train_paths, val_paths
        )

        self.device = device
        self.args = args
        self.fold = fold
        self.beta = beta

    def load_dataloader(self, train_paths, val_paths):
        train_dataset = VideoDataset(train_paths)
        val_dataset = VideoDataset(val_paths)

        train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )
        val_dataloader = data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        return train_dataloader, val_dataloader

    def load_baselines(self, paths):
        baselines = {}
        for path in paths:
            id = get_filename(path)
            baselines[id] = 0.0
        return baselines

    def save_policy(self):
        self.policy.eval()
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"models/{self.args.run_name}_fold{self.fold}.pth",
        )
        self.policy.train()

    def load_policy(self, path):
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def learn(self, epochs, num_episodes):
        for epoch in range(epochs):
            losses = AverageMeter()

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for idx, (feature, id) in enumerate(pbar):
                id = id[0]
                feature = feature.to(self.device)
                probs = self.policy(feature)

                loss = -self.beta * (probs.mean() - 0.5) ** 2
                distr = Bernoulli(probs)
                rewards = AverageMeter()

                for _ in range(num_episodes):
                    actions = distr.sample()
                    log_probs = distr.log_prob(actions)
                    reward = compute_reward(
                        feature, actions, use_gpu=True, temp_dist_thre=40
                    )
                    loss = loss - log_probs.mean() * (reward - self.baselines[id])
                    rewards.update(reward.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.baselines[id] = 0.9 * self.baselines[id] + 0.1 * rewards.avg

                losses.update(loss.item())
                pbar.set_postfix(loss=losses.avg, reward=rewards.avg)
                logging.info(
                    "Epoch: {}, Step: {}, Loss: {}, Reward: {}".format(
                        epoch + 1, idx + 1, loss.item(), rewards.avg
                    )
                )

            self.save_policy()


for fold in range(5):
    f = open(f"folds/fold_{fold}.json")
    dataset = json.load(f)
    f.close()
    train_paths = dataset["train"]
    val_paths = dataset["val"]
    agent = REINFORCE(
        train_paths=train_paths, val_paths=val_paths, args=args, fold=fold,
    )
    logging.info(f"Fold: {fold}")
    print(f"Fold: {fold}")
    agent.learn(args.epochs, args.num_episodes)
    logging.info("--------------------------------------")
    print("--------------------------------------")

