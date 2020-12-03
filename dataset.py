import os
import torch
import h5py
import json
from torch.utils import data
from utils import get_filename
import pandas as pd
import numpy as np


class VideoDataset(data.Dataset):
    def __init__(self, paths, cnn_feat="resnet50"):
        super().__init__()
        self.paths = paths
        self.cnn_feat = cnn_feat
        self.df = pd.read_csv(
            "ydata-tvsum50-anno.tsv",
            sep="\t",
            header=None,
            names=["id", "category", "scores"],
        )

        f = open("id_to_key_map.json")
        self.id_key_map = json.load(f)
        f.close()

        self.dataset = h5py.File("datasets/eccv16_dataset_tvsum_google_pool5.h5", "r")

        self.change_points = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["change_points"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.nfps = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frame_per_seg"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.picks = dict(
            zip(
                list(self.dataset.keys()),
                [self.dataset[key]["picks"][...] for key in list(self.dataset.keys())],
            )
        )

        self.features = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["features"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.user_summary = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["user_summary"][...]
                    for key in list(self.dataset.keys())
                ],
            )
        )

        self.n_frames = dict(
            zip(
                list(self.dataset.keys()),
                [
                    self.dataset[key]["n_frames"][()]
                    for key in list(self.dataset.keys())
                ],
            )
        )

    def __getitem__(self, i):
        path = self.paths[i]
        id = get_filename(path)

        key = self.id_key_map[id]
        user_summary = self.user_summary[key]
        change_points = self.change_points[key]
        nfps = self.nfps[key]
        picks = torch.LongTensor(self.picks[key])
        n_frames = self.n_frames[key]

        if self.cnn_feat == "resnet50":
            feature = torch.load(path)
            feature = feature[picks, :]
        else:
            feature = self.features[key]

        return {
            "feature": feature,
            "user_summary": user_summary,
            "id": id,
            "change_points": change_points,
            "nfps": nfps,
            "picks": picks,
            "n_frames": n_frames,
        }

    def __len__(self):
        return len(self.paths)
