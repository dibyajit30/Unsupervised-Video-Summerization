# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from PolicyGradient import PolicyNet

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def videoFrames(filename):
    video = cv2.VideoCapture(filename)
    frames = []
    total_frames = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        total_frames += 1
        frames.append(frame)
    video.release()
    
    selected_frames = []
    for i in range(0, total_frames, 15):
        selected_frames.append(frames[i])
    frames = np.stack(selected_frames)
    return frames

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.to(device)
resnet.eval()

def features(frames):
    features = []
    inputs = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        frame = frame.unsqueeze(0).to(device)
        inputs.append(frame)
        if len(inputs) % batch_size == 0:
            inputs = torch.cat(inputs, 0)
            with torch.no_grad():
                feat = resnet(inputs)
            features.append(feat.squeeze().cpu())
            inputs = []
    
    if len(inputs) > 0:
        inputs = torch.cat(inputs, 0)
        with torch.no_grad():
            feat = resnet(inputs)
        features.append(feat.squeeze(-1).squeeze(-1).cpu())
    
    features = torch.cat(features, 0)
    features = features.view(-1, 2048)
    return features

class Arguments:
    def __init__(self, cnn_feat='resnet50', train_cnn=False):
        self.cnn_feat = cnn_feat
        self.train_cnn = train_cnn

def summaryIndex(features):
    args = Arguments()
    policy = PolicyNet(args=args)
    ckpt = torch.load('lstm_model.pth', map_location=device)
    policy.load_state_dict(ckpt["model"])
    
    probs, _ = policy(features.unsqueeze(0))
    probs = probs.squeeze()
    
    selected_frame_index = torch.topk(probs, k=probs.size(0)//2).indices
    selected_frame_index = torch.sort(selected_frame_index).values
    
    return selected_frame_index

def summary(frames, selected_frame_index, width, height, filename='summary.mp4'):
    selected_frame_index = list(selected_frame_index)
    summary_frames = frames[[selected_frame_index]]
    
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), 10, (width, height))
    for frame in summary_frames:
        out.write(frame)
  
def summarize(filename, summary_filename):      
    frames = videoFrames(filename)
    video_features = features(frames)
    indices = summaryIndex(video_features)
    summary(frames, indices, frames.shape[2], frames.shape[1], filename=summary_filename)