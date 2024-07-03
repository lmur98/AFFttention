import torch
import json
import os
import numpy as np
import tqdm
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/lmur/stillfast_baseline/stillfast/')
from stillfast.datasets.ek_55_sta_dataset import EpicKitchens_ShortTermAnticipation


class EK55_Propagate_Aff(EpicKitchens_ShortTermAnticipation):
    def __init__(self):
        super().__init__(split = 'val')
        
        print(self.verb_classes)
        self.nouns_dict = self.noun_classes
        self.verbs_dict = self.verb_classes
        
        self.n_verbs = len(self.verbs_dict)
        self.n_nouns = len(self.nouns_dict)   
        print('The number of verbs and nouns are:', self.n_verbs, self.n_nouns) 

        self.cmap_n = plt.get_cmap('tab20b')(np.linspace(0, 1, len(self.nouns_dict)))
        self.cmap_v = plt.get_cmap('tab20c')(np.linspace(0, 1, len(self.verbs_dict)))

        self.narrations_dir = '/home/lmur/hum_obj_int/narrations_dict'

    def __getitem__(self, idx):
        sample = self._annotations['annotations'][idx]

        #Load the STA labels
        sta_verbs, sta_nouns = [], []
        n_obj = len(sample['objects']['noun_category_id'])
        for obj in range(n_obj):
            sta_verbs.append(sample['objects']['verb_category_id'][obj])
            sta_nouns.append(sample['objects']['noun_category_id'][obj])
        frame_number = round(sample['timestamp'] * 30) + 2

        return {'v_id': sample['video_id'], 
                'frame': frame_number,
                'sta_verb': sta_verbs, 
                'sta_noun': sta_nouns}

if __name__ == '__main__':
    dataset = EK55_Propagate_Aff()
    print(dataset[0])