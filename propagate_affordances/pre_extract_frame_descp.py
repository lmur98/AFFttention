import torch
import numpy as np

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

import clip
import os
import tqdm
from propagate_AFF_dataset import Ego4D_Propagate_Aff


class Frame_descriptor_Extractor():
    def __init__(self):
        self.CLIP_model, self.CLIP_transform = clip.load("ViT-L/14", device='cuda')
        self.still_frames_path = '/home/furnari/data/ego4d/v2-15-02-23/object_frames/'
        self.output_dir = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_VAL/CLIP_features_per_frame_v2'

    ## -------------------------------Extract a CLIP descriptor from a single frame ---------------------------#
    def save_frame_descriptors(self, frame):
        frame_desc = self.extract_frame_descriptors(frame)
        np.save(os.path.join(self.output_dir, f"{frame['v_id']}_{frame['frame']:07d}.npy"), frame_desc)

    def extract_frame_descriptors(self, frame):
        print(frame['v_id'], frame['frame'])
        if len(frame['text']) > 0:
            frame_text_CLIP = self.text_to_CLIP_features(frame['text'][0])
        else:
            print('There are not narration!!')
            frame_text_CLIP = None
        frame_visual_CLIP = self.img_to_CLIP_features(frame)
            
        return {'text': frame_text_CLIP, 'visual': frame_visual_CLIP}

    def img_to_CLIP_features(self, frame):
        img = Image.open(os.path.join(self.still_frames_path, f"{frame['v_id']}_{frame['frame']:07d}.jpg")).convert('RGB')
        #imgs = [self.video_dataset.get(v_id, frame).convert('RGB') for frame in frames] 
        img = self.CLIP_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            embeddings = self.CLIP_model.encode_image(img)
        return embeddings.cpu()

    def clean_narration(self, sentence):
        sentence = sentence.replace("#C ", "")
        sentence = sentence.replace("#Unsure", "")
        sentence = sentence.replace("#", "")
        return sentence

    def text_to_CLIP_features(self, narrations):
        if 'narrator_1' not in narrations.keys() and 'narrator_2' not in narrations.keys():
            return None
        elif 'narrator_1' not in narrations.keys():
            sentence = self.clean_narration(narrations['narrator_2'])
        elif 'narrator_2' not in narrations.keys():
            sentence = self.clean_narration(narrations['narrator_1'])
        else:
            sentence = self.clean_narration(narrations['narrator_1']) + ' and ' + self.clean_narration(narrations['narrator_2'])
        print('The input sequence', sentence)
        inputs = clip.tokenize([sentence]).to('cuda')
        with torch.no_grad():
            embeddings = self.CLIP_model.encode_text(inputs)
        return embeddings.cpu()

val_dataset = Ego4D_Propagate_Aff()
desc_extractor = Frame_descriptor_Extractor()

for i in tqdm.tqdm(range(len(val_dataset))):
    frame = val_dataset[i]
    frame_desc = desc_extractor.save_frame_descriptors(frame)
    print()
    #print(frame_desc)
    #break





