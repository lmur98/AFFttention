import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List

import sys
sys.path.append('/home/lmur/stillfast_baseline/stillfast/')
from stillfast.datasets.sta_hlmdb import Ego4DHLMDB
import stillfast.datasets.ek_55_sta_dataset
from stillfast.datasets.ek_55_sta_dataset import EpicKitchens_ShortTermAnticipation, EK55_HLMDB_STA_Still_Video
from stillfast.config.defaults import get_cfg
from torchvision import transforms
import time
from itertools import chain
import cv2
import pandas as pd



class EK55_Environmental_Affordances(EpicKitchens_ShortTermAnticipation):
    """
    Ego4d Short Term Anticipation StillVideo Dataset
    """

    def __init__(self, cfg, split):
        super(EK55_Environmental_Affordances, self).__init__(cfg, split)
        self._fast_EK55_hlmdb = EK55_HLMDB_STA_Still_Video('/ssd/furnari/ek100/', readonly=True, lock=False)
        
        self.frames_per_clip = 15
        self.STA_annots = self._annotations['annotations']
        print(len(self.STA_annots), 'sta and aff')
        self.unique_video_ids = self.pickle_file['video_id'].unique()
        print(self.unique_video_ids)
        
    def __len__(self): #We build a topological graph per video, not per scene!!
        return len(self.unique_video_ids)
    
    def _load_img_low_res(self, video_id, fast_frames_list):
        """ Load frames from video_id and frame_number """
        new_frames_list = []
        for f in fast_frames_list:
            new_frames_list.append(f"{video_id}_frame_{f:010d}.jpg")
        low_res_frame = self._load_EK55_frames_lmdb('rgb', new_frames_list)
        low_res_frame = low_res_frame[0].convert('RGB')
        return low_res_frame
    
 
    def _transform_egotopo(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
        return transform
    
    def from_STA_annot_to_narration(self, frame_annotated):
        n_objects = len(frame_annotated['objects']['noun_category_id'])
        narration = ''
        for n in range(n_objects):
            verb = self.verb_classes[frame_annotated['objects']['verb_category_id'][n]]
            noun = self.noun_classes[frame_annotated['objects']['noun_category_id'][n]]
            narration += verb + ' the ' + noun + ' and '
        narration = narration[:-5]
        return narration

    def _load_FULL_VIDEO_annotations(self, idx):
        """ Load annotations for the idx-th video """
        video_idx = self.unique_video_ids[idx]
        main_frames_list = []
        video_annotations = []
        narrations = []
        for frame_annotated in self.STA_annots:
            if frame_annotated['video_id'] == video_idx:
                fast_time_number = round(frame_annotated['timestamp'] * 30) + 2
                main_frames_list.append(fast_time_number)
                video_annotations.append(frame_annotated['objects'])
                narrations.append(self.from_STA_annot_to_narration(frame_annotated))
               
        return video_idx, main_frames_list, video_annotations, narrations


    def __getitem__(self, idx):
        """ Get the idx-th full video, composed of several annotations"""
        video_id, main_frames_list, all_video_annotations, narrations = self._load_FULL_VIDEO_annotations(idx)   
    
        return {'video_id': video_id, 
                'frames_list': main_frames_list, #List of frames [int]
                'targets': all_video_annotations,
                'narrations': narrations}

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file('/home/lmur/stillfast_baseline/stillfast/output/sta/StillFast_Fullmodel_Ego4D_v2/version_1/config.yaml')
    dataset = EK55_Environmental_Affordances(cfg, 'train')
    sample = dataset[0]
   
        