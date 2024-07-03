import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata
import numpy as np
import os
import time
import cv2 as cv
import itertools
import tqdm
from joblib import Parallel, delayed
import cv2
import json
import lmdb
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/lmur/hum_obj_int/stillfast')
import networkx as nx
from stillfast.datasets.ego4d_training_Similarity_frames import Ego4D_training_Similarity_frames
from stillfast.config.defaults import get_cfg #Import the configurations
from ego_topo.build_graph.tools.superpoint.model import SuperPointFrontend
from ego_topo.build_graph.tools.generate_sp_matches import get_pair_homography
from sklearn.metrics.pairwise import euclidean_distances
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video

class Draw_Local_Graph():
    def __init__(self, split):
        self.split = split
        if self.split == 'train':
            self.root = '/home/lmur/catania_lorenzo/v2_data_extracted/output_topo_graphs_TRAIN'
        elif self.split == 'val':
            self.root = os.path.join('/home/lmur/hum_obj_int/stillfast/v2_output_topo_graphs_VAL', 'local_topological_graphs')
        self.json_files = os.path.join(self.root, 'local_topological_graphs_REVIEWED')
        self.output_dir = os.path.join(self.root, 'visualize_reviewed')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self._fast_hlmdb = Ego4DHLMDB_STA_Still_Video('/ssd/furnari/sta_lmdb_v2', readonly=True, lock=False)
    
        
    def draw(self):
        for file in os.listdir(self.json_files):
            v_id = file.split('.')[0].split('_')[-1]

            if file.endswith('.json'):
                print('Processing', file)
                data = json.load(open(os.path.join(self.json_files, file), 'r'))
                print(data)
                #Create folder to save
                folder_name = file.split('.')[0].split('_')[-1]
                v_id = folder_name
                output_dir = os.path.join(self.output_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                self.draw_local(data, output_dir, v_id)

    def draw_local(self, local_graph, output_dir, v_id):
        for n, node in enumerate(local_graph):
            if len(node['frames_in_node']) > 12:
                random_choice = np.random.choice(len(node['frames_in_node']), 12, replace=True)
                fig, axs = plt.subplots(3, 4)
            else:
                random_choice = np.random.choice(len(node['frames_in_node']), 6, replace=True)
                fig, axs = plt.subplots(3, 2)
            
            
            frames = [node['frames_in_node'][i] for i in random_choice]
            #texts = [node['narrator_1'][i][5:] for i in random_choice]
            imgs = [self._fast_hlmdb.get(v_id, frame).convert('RGB') for frame in frames]
            print(imgs)
            
            for i in range(3):
                axs[i, 0].imshow(imgs[i])
                axs[i, 0].axis('off')
                axs[i, 1].imshow(imgs[i+3])
                axs[i, 1].axis('off')
                if len(node['frames_in_node']) > 12:
                    axs[i, 2].imshow(imgs[i+6])
                    axs[i, 2].axis('off')
                    axs[i, 3].imshow(imgs[i+9])
                    axs[i, 3].axis('off')
                
            """
            for i in range(3):
                # Primera columna: Texto
                axs[i, 0].text(0.5, 0.7, texts[i][0:40], ha='center', va='center', fontsize=8)
                axs[i, 0].text(0.5, 0.3, texts[2*i][0:40], ha='center', va='center', fontsize=8)
                axs[i, 0].axis('off')
    
                # Segunda columna: Imagen
                axs[i, 1].imshow(imgs[2*i])
                axs[i, 1].axis('off')
    
                # Tercera columna: Imagen con título
                axs[i, 2].imshow(imgs[2*i + 1])
                axs[i, 2].axis('off')
            """

            fig.suptitle('Node: ' + str(n) + ' with ' + str(len(node['frames_in_node'])) +' frames showing', fontsize = 10)
            plt.tight_layout(pad = 0.5)
            plt.savefig(os.path.join(output_dir, str(node['node_name']) + '.png'))
            plt.close()

draw_class = Draw_Local_Graph(split='train')
draw_class.draw()
