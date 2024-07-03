# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as tmodels
import torchvision.transforms as transforms
import os
from PIL import Image, ImageOps
import sys
sys.path.append('/home/lmur/hum_obj_int/stillfast')
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video

class SiameseR18_5MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.trunk = tmodels.resnet18(pretrained=True)
        self.trunk.fc = nn.Sequential()
        self.compare = nn.Sequential(
                            nn.Linear(512*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2))

    def forward(self, batch, softmax=False):

        featA, featB = self.trunk(batch['imgA']), self.trunk(batch['imgB']) 
        featAB = torch.cat([featA, featB], dim=1)
        sim_pred = self.compare(featAB)

        loss_dict = {}
        if 'label' in batch: #In training it returns only the loss. The Loss uses the LOGITS
            loss = F.cross_entropy(sim_pred, batch['label'], reduction='none')
            loss_dict.update({'sim': loss})

        if softmax: #In inference it applies a Softmax and returns the second class (positive class)
            sim_pred = F.softmax(sim_pred, 1)[:, 1]
        
        return sim_pred, loss_dict


class R18_5MLP(SiameseR18_5MLP):
    def __init__(self):
        super().__init__()
        self.compare = nn.Sequential(
                            nn.Linear(512*2, 512),
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2))
        
        self.trunk = nn.Identity()

"""

video_dataset = Ego4DHLMDB_STA_Still_Video('/ssd/furnari/sta_lmdb/', readonly=True, lock=False)

def load_img(video_id, frame_n):
    img = video_dataset.get(video_id, frame_n)
    return img.convert('RGB')

def transform_function():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(256), #Downscales the image in a x4 factor aprox
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    return transform

v_id = '0836e1a4-11e6-4b31-bd39-f8e083fdadb3'
transform = transform_function()
checkpoint = torch.load('/home/lmur/hum_obj_int/stillfast/extract_affordances/saved_models/Ego_4D_Siamese_R18_5MLP_34_6528.pth', map_location='cpu')

img1 = transform(load_img(v_id, 19861)).unsqueeze(0)
img2 = transform(load_img(v_id, 19816)).unsqueeze(0)

model = SiameseR18_5MLP()
model.eval()
model.load_state_dict(checkpoint['net'])

with torch.no_grad():
    out = model({'imgA': img1, 'imgB': img2}, softmax=True)
    print(out)
"""

