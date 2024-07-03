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
   
        
        self.AFF_labels_json = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_VAL/EGO4D_AFF_REVIEW_VAL/AFF_EGO4D_V1_VAL_complete_d030.json'
        self.STA_labels_json = '/home/furnari/ego4d_data/v1/annotations/fho_sta_val.json'
        self.STA_labels = json.load(open(self.STA_labels_json, 'r'))['annotations']
        self.AFF_labels = json.load(open(self.AFF_labels_json, 'r'))
        print(len(self.STA_labels), len(self.AFF_labels), 'sta and aff')

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
        return len(self.AFF_labels)
    
    #MODIFYYYYY
    def from_STA_to_ids(self, STA_interaction):
        STA_interaction = STA_interaction.replace('vegetable_fruit', 'vegetable')
        STA_interaction = STA_interaction.replace('playing_cards', 'playing')
        STA_interaction = STA_interaction.replace('tape_measure', 'tape')
        STA_interaction = STA_interaction.replace('rubber_band', 'rubber')
            
        sta_verb = STA_interaction.split('_')[:-1]
        sta_verb = '_'.join(sta_verb)
        sta_noun = STA_interaction.split('_')[-1:][0]

        verb_id = [d['id'] for d in self.verbs_dict if d['name'] == sta_verb][0]
        noun_id = [d['id'] for d in self.nouns_dict if d['name'] == sta_noun][0]
        return verb_id, noun_id

    def get_STA_from_AFF(self, sample):
        #Get the STA with video_id == sample['v_id'] and frame == sample['frame']
        for STA_sample in self.STA_labels:
            if STA_sample['video_id'] == sample['v_id'] and STA_sample['frame'] == sample['frame']:
                sta_v, sta_n = [], []
                for interaction in STA_sample['objects']:
                    #print(interaction, 'the STA interaction is')
                    sta_v.append(interaction['verb_category_id'])
                    sta_n.append(interaction['noun_category_id'])
                return sta_v, sta_n


    def __getitem__(self, idx):
        sample = self.AFF_labels[idx]
        #Load the STA labels
        sta_verb, sta_noun = self.get_STA_from_AFF(sample)
        #Load the AFF labels
        aff_noun = np.array(sample['aff_nouns']).astype(float) #(87,)
        aff_verb = np.array(sample['aff_verbs']).astype(float) #(74,) 
        aff_noun[aff_noun > 0] = 1
        aff_verb[aff_verb > 0] = 1
        
        #Load the narrations
        narrations = self.read_narrations(sample['v_id'], sample['frame'])
        #print('the narrations are', narrations)

       
        return {'v_id': sample['v_id'], 
                'frame': sample['frame'],
                'text': narrations,
                'aff_verb': aff_verb, 'aff_noun': aff_noun, 
                'sta_verb': sta_verb, 'sta_noun': sta_noun}