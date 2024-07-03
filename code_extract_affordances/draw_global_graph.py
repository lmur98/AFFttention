import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as tmodels
import torchvision.transforms as transforms
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/lmur/hum_obj_int/stillfast')
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video


video_dataset = Ego4DHLMDB_STA_Still_Video('/ssd/furnari/sta_lmdb/', readonly=True, lock=False)
json_file = '/home/lmur/hum_obj_int/stillfast/output_topo_graphs/clusters_prueba.json'
clusters = json.load(open(json_file, 'r'))
outpur_dir = '/home/lmur/hum_obj_int/stillfast/output_topo_graphs/visualize_clusters_prueba'
if not os.path.exists(outpur_dir):
    os.makedirs(outpur_dir)

for n, cluster in enumerate(clusters):

    subclusters = [cluster[i:i+8] for i in range(0, len(cluster), 8)]
    for n2, subcluster in enumerate(subclusters):
        num_columns = len(subcluster)
        num_rows = 4
        v_ids_list = []
        frames_list = []
        texts_list = []
        for node in subcluster:
            node_v_id = node['v_id']
            random_choice = np.random.choice(len(node['text']), 4)
            print(random_choice)
            print(len(node['visits']), len(node['text']))
            node_frames = [node['visits'][i] for i in random_choice]
            node_texts = [node['text'][i] for i in random_choice]
            for frame, text in zip(node_frames, node_texts):
                v_ids_list.append(node_v_id)
                frames_list.append(frame)
                texts_list.append(text)

    

        #In each cell, we will put an image
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(num_columns, num_rows), dpi = 300)
        for i in range(num_rows):
            for j in range(num_columns):
                index = j * num_rows + i
                if num_columns == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]
                img = video_dataset.get(v_ids_list[index], frames_list[index]).convert('RGB')
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(str(texts_list[index][0:30]), fontsize = 4)
        
        #Put a global title:
        fig.suptitle('Cluster: ' + str(n) + ' formed by ' + str(len(cluster)) + ' nodes', fontsize=16)
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(outpur_dir, 'cluster' + str(n) + '_' + str(n2) + '.png'), dpi = 300)
