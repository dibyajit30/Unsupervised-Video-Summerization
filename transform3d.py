# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

"""
   Copied from: https://github.com/pytorch/vision/blob/master/references/video_classification/transforms.py
"""
class ConvertBHWCtoBCHW(nn.Module):
    """Convert tensor from (B, H, W, C) to (B, C, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)
