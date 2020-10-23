import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beroulli
from torch.utils import data
import json
import logging


class VideoDataset(data.Dataset):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def __getitem__(self, i):
        path = self.paths[i]
        feature = torch.load(path)
        return feature

    def __len__(self):
        return len(self.paths)


class RolloutBuffer:
    def __init__(self, rollout_steps, gamma, device):
        self.rollout_steps = rollout_steps
        self.gamma = gamma
        self.device = device
        self.states = None
        self.rewards = None
        self.actions = None
        self.count = None
        self.reset()

    def reset(self):
        self.states = [None] * self.rollout_steps
        self.rewards = [None] * self.rollout_steps
        self.actions = [None] * self.rollout_steps
        self.count = 0

    def store(self, state, reward, action):
        self.states[self.count] = state
        self.rewards[self.count] = (self.gamma ** self.count) * reward
        self.actions[self.count] = action
        self.count += 1

    def compute_returns(self):
        returns = []
        for i in range(self.count):
            returns.append(sum(self.rewards[i : self.count]) / self.gamma ** i)
        return returns

    def get_values(self):
        states = torch.stack(self.states[: self.count]).to(self.device)
        actions = torch.tensor(self.actions[: self.count]).to(self.device).long()
        returns = self.compute_returns()
        returns = torch.stack(returns).to(self.device)
        self.reset()
        return states, returns, actions


def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Copied from: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.0)
        if use_gpu:
            reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.0)
        if use_gpu:
            reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1.0 - torch.matmul(
            normed_seq, normed_seq.t()
        )  # dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.0
        reward_div = dissim_submat.sum() / (
            num_picks * (num_picks - 1.0)
        )  # diversity reward [Eq.3]

    # compute representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:, pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())  # representativeness reward [Eq.5]

    # combine the two rewards
    reward = (reward_div + reward_rep) * 0.5

    return reward


class PolicyNet(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=256, num_layers=2, dropout=0.2):
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

