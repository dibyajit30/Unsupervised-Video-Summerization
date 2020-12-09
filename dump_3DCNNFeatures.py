# -*- coding: utf-8 -*-
from tqdm.auto import tqdm
import numpy as np
import glob
import os
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import gc
from torchvision import transforms as T
from transform3d import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW

device = "cuda" if torch.cuda.is_available() else "cpu"

normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
preprocess = transforms.Compose([
        ConvertBHWCtoBCHW(),
        T.ConvertImageDtype(torch.float32),
        T.Resize((128, 171)),
        T.RandomHorizontalFlip(),
        normalize,
        T.RandomCrop((112, 112)),
        ConvertBCHWtoCBHW()
])

resnet = models.video.r3d_18(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.to(device)
resnet.eval()

files = glob.glob("videos_npy/*.npy")
errors = []
for i, file in enumerate(files):
    prefix = file.split("/")[-1].split(".")[0]
    save_path = f"cnn_feats/{prefix}.pt"
    if os.path.exists(save_path):
        continue
    try:
        images = np.load(file)
        images = torch.Tensor(images)
        images = preprocess(images)
        images = images.unsqueeze(0)
    except:
        errors.append(file)
        continue
    
    features = resnet(images)
    
    
    features = features.view(-1, 512)
    torch.save(features.cpu(), save_path)
    del features
    gc.collect()

print("Errors")
print(errors)

