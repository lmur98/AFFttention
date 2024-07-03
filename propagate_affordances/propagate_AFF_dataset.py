import torch
import json
import os
import numpy as np
import tqdm
from matplotlib import pyplot as plt

class Ego4D_Propagate_Aff(torch.utils.data.Dataset):
    def __init__(self):
        #prev_nouns_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_nouns_list.json', 'r')) #V1
        prev_nouns_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_v2_nouns_list.json', 'r'))
        self.nouns_dict = []
        for noun in prev_nouns_dict: 
            self.nouns_dict.append({'id': noun['id'], 'name': noun['name'].split('_')[0]}) 
        #prev_verbs_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_verbs_list.json', 'r'))
        prev_verbs_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_v2_verbs_list.json', 'r'))
        self.verbs_dict = []
        for verb in prev_verbs_dict:
            self.verbs_dict.append({'id': verb['id'], 'name': verb['name'].split('_(')[0]})
        self.n_verbs = len(self.verbs_dict)
        self.n_nouns = len(self.nouns_dict)   
        print('The number of verbs and nouns are:', self.n_verbs, self.n_nouns) 

        self.cmap_n = plt.get_cmap('tab20b')(np.linspace(0, 1, len(self.nouns_dict)))
        self.cmap_v = plt.get_cmap('tab20c')(np.linspace(0, 1, len(self.verbs_dict)))
   
        self.split = 'val'
        self.STA_labels_json = '/home/furnari/ego4d_data/v2/annotations/fho_sta_val.json'
        self.STA_labels = json.load(open(self.STA_labels_json, 'r'))['annotations']
        print(len(self.STA_labels), 'sta and aff')

        self.narrations_dir = '/home/lmur/hum_obj_int/narrations_dict'

    def read_narrations(self, video_id, frame):
        narrations_json = json.load(open(os.path.join(self.narrations_dir, f"{video_id}.json")))
        
        if 'narration_pass_1' in narrations_json:
            clip_narrations_1 = narrations_json['narration_pass_1']['narrations']
            narrator_1 = True
        else:
            narrator_1 = False
        if 'narration_pass_2' in narrations_json:
            clip_narrations_2 = narrations_json['narration_pass_2']['narrations']
            narrator_2 = True
        else:
            narrator_2 = False
        
        #We need to find the narration that corresponds to each frame
        frames_narrations = []
        if not(narrator_1) and not(narrator_2):
            return frames_narrations
        if narrator_1:
            narration_1_empty = True
            for n in range(len(clip_narrations_1) - 1):
                narration = clip_narrations_1[n]['narration_text']
                start_frame = clip_narrations_1[n]['timestamp_frame']
                end_frame = clip_narrations_1[n + 1]['timestamp_frame']
                if frame >= start_frame and frame < end_frame:
                    to_append = {'frame_number': frame, 'begin_narration': start_frame, 'end_narration': end_frame, 'narrator_1': narration}
                    narration_1_empty = False
                    break

        if narrator_2:
            narration_2_empty = True
            for n in range(len(clip_narrations_2) - 1):
                narration = clip_narrations_2[n]['narration_text']
                start_frame = clip_narrations_2[n]['timestamp_frame']
                end_frame = clip_narrations_2[n + 1]['timestamp_frame']
                if frame >= start_frame and frame < end_frame:
                    narration_2_empty = False
                    if not(narrator_1) or narration_1_empty:
                        to_append = {'frame_number': frame, 'begin_narration': start_frame, 'end_narration': end_frame}
                    to_append.update({'narrator_2': narration})
                    break
        if narration_1_empty and narration_2_empty:
            to_append = {'frame_number': frame, 'begin_narration': None, 'end_narration': None}
        frames_narrations.append(to_append)
        return frames_narrations

    def __len__(self):
        return len(self.STA_labels)
    
    def __getitem__(self, idx):
        sample = self.STA_labels[idx]
        if self.split == 'val':
            #Load the STA labels
            sta_verbs, sta_nouns = [], []
            for obj in sample['objects']:
                sta_verbs.append(obj['verb_category_id'])
                sta_nouns.append(obj['noun_category_id'])
            return {'v_id': sample['video_uid'], 
                    'frame': sample['frame'],
                    #'text': narrations,
                    'sta_verb': sta_verbs, 'sta_noun': sta_nouns}
        else:
            return {'v_id': sample['video_uid'], 'frame': sample['frame']}