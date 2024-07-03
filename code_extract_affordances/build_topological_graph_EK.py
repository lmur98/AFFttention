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

import networkx as nx
import sys
sys.path.append('/home/lmur/stillfast_baseline/stillfast')
from stillfast.config.defaults import get_cfg #Import the configurations
import torchvision.transforms as transforms

#from ego_topo.build_graph.localization_network.model import SiameseR18_5MLP, R18_5MLP
from Siamesse_NN import SiameseR18_5MLP, R18_5MLP
from ek55_env_aff_dataset import EK55_Environmental_Affordances
import random
#---------Goal: Take a video sequence and extract the visually similar frames---------#
#---------When two frames are similar: they belong to the same node and they share the actions -> affordances---------#

class Create_Local_Topo_Graph():
    def __init__(self, video_sequence, dataset, viz = False, frame_inp = True, locnet_wts = None):
        self.v_id = video_sequence['video_id']
        self.frames_list = video_sequence['frames_list']
        self.narrations = video_sequence['narrations']
        self.STA_annots = video_sequence['targets']
        print('Len before', len(self.frames_list), len(self.narrations), len(self.STA_annots))

        self.sample_ratio = 14 #6 fps
        self.dataset = dataset
        self.thresh_upper = 0.7
        self.thresh_lower = 0.4 #We want to be very sure that the frame is not similar to any other
        
        self.args_viz = viz
        self.frame_inp = frame_inp
        #self.locnet_wts = '/home/lmur/hum_obj_int/stillfast/extract_affordances/saved_models/Ego_4D_Siamese_R18_5MLP_34_6528.pth'
        self.locnet_wts = '/home/lmur/catania_lorenzo/EK_data_extracted/ckpt_E_250_I_19250_L_0.368_A_0.834.pth'
        self.output_dir = '/home/lmur/catania_lorenzo/EK_data_extracted/output_topo_graphs_TRAIN'
        self.transform = self._transform_egotopo()

        #Load the localization network
        self.net = SiameseR18_5MLP()
        checkpoint = torch.load(self.locnet_wts, map_location = 'cpu')
        self.net.load_state_dict(checkpoint['net'])
        self.net.cuda('cuda:3').eval() #MOD FOR CPU!!

        #Define the frames
        self.window_size = 3
        self.start_frame = self.frames_list[0]
        self.end_frame = self.frames_list[-1]
        self.frames = [(self.v_id, f_id) for f_id in self.frames_list] #List with all the extracted frames from the video
        self.frame_to_idx = {frame: idx for idx, frame in enumerate(self.frames)}
        print(f'Generating graph for {self.v_id}. {len(self.frames)} total frames')

    def _transform_egotopo(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
        return transform

    def create_new_node(self, frame_number, frame_narrations, frame_STA_annots):
        node = {'node_name': frame_number, #We give the name of the first frame that sees
                'frames_in_node': [], #Tuple with video_id, frame_number
                'narrator': [], #List of the narrations
                'STA_noun': [], 'STA_verb': []} #List of the STA descriptions  
        node['frames_in_node'].append(frame_number)
        node['narrator'].append(frame_narrations)
        n_annots = len(frame_STA_annots['noun_category_id'])
        for n in range(n_annots):
            node['STA_noun'].append(frame_STA_annots['noun_category_id'][n])
            node['STA_verb'].append(frame_STA_annots['verb_category_id'][n])
        self.G.append(node)

    #Change this
    def save(self):
        with open(os.path.join(self.output_dir, 'local_topological_graphs_COMPLETE', 'topo_graph_' + self.v_id + '.json'), 'w') as f:
            json.dump(self.G, f)

    def load_frame(self, frame):
        frame = self.dataset._load_img_low_res(self.v_id, [frame]) #Load as PIL and read as RGB
        if frame is None:
            return None
        frame = self.transform(frame) #Same transform as the training
        return frame
        
    def merge_frame_in_node(self, top1_node_name, frame_number, frame_narrations, frame_STA_annots):
        for node in self.G:
            if node['node_name'] == top1_node_name:
                node['frames_in_node'].append(frame_number)
                node['narrator'].append(frame_narrations)
                n_annots = len(frame_STA_annots['noun_category_id'])
                for n in range(n_annots):
                    node['STA_noun'].append(frame_STA_annots['noun_category_id'][n])
                    node['STA_verb'].append(frame_STA_annots['verb_category_id'][n])
                return

    def divide_members_in_visits(self, node_members):
        #Split the list if the difference between two consecutive frames is bigger than 1
        visits = []
        visit = []
        for i in range(len(node_members)):
            if i == 0:
                visit.append(node_members[i])
            else:
                if node_members[i] - node_members[i-1] > self.sample_ratio:
                    visits.append(visit)
                    visit = []
                visit.append(node_members[i])
        visits.append(visit)
        return visits

    def score_nodes(self, frame_i):
        scores = []
        
        for node in self.G:
            node_members = node['frames_in_node']
            visits = self.divide_members_in_visits(node_members)
            key_frames = []
            for n, visit in enumerate(visits):
                frames = list(range(visit[0], visit[-1] + 1)) #List of frames in the visit (not sampled)
                if len(frames) < self.window_size:
                    key_frames += frames
                else: #Select the center frame for each visit
                    mid = len(frames) // 2
                    key_frames += frames[mid - self.window_size // 2: mid + self.window_size // 2]
            if len(key_frames) > 10:
                #Select randomly 20 frames
                key_frames = random.sample(key_frames, 10)
            window = [frame_i - 10, frame_i] #Window around the frame
            score = self.score_pair_sets(window, key_frames)
            scores.append({'node_name': node['node_name'], 'score': score})
        return scores

    def score_pair_sets(self, set1, set2):
        if len(set1)==0 or len(set2)==0:
            return -1
        S = []
        for i in set1:
            for j in set2:
                S.append(self.pair_scores_VISUAL(i, j))
        return np.mean(S)

    def pair_scores_TIME(self, frame1, frame2):
        sim = 1 - np.abs(frame1 - frame2) / 1000
        return sim

    def pair_scores_VISUAL(self, fA, fB):
        if fB < fA:
            fA, fB = fB, fA
        featA = self.load_frame(fA)
        featB = self.load_frame(fB)
        if featA is None or featB is None:
            return 0    
        featA = featA.unsqueeze(0).cuda('cuda:3') #MOD FOR CPU!!
        featB = featB.unsqueeze(0).cuda('cuda:3') #MOD FOR CPU!!
        with torch.no_grad():
            sim = self.net({'imgA':featA, 'imgB':featB}, softmax=True)
        sim = sim[0].item()
        return sim

    def build(self):
        potential_path = os.path.join('/home/lmur/catania_lorenzo/v2_data_extracted/output_topo_graphs_TRAIN/local_topological_graphs_COMPLETE', 'topo_graph_' + self.v_id + '.json')
        if os.path.exists(potential_path):
            print('The graph already exists')
            return
    
        self.G = [] #List of dicts. Each dict is a node with its features
        progress_bar = tqdm.tqdm(total=len(self.frames_list))
        for f in range(len(self.frames_list)):
            frame_number = self.frames_list[f]

            progress_bar.update()
            frame_number = self.frames_list[f]
            frame_narrations = self.narrations[f]
            frame_STA_annots = self.STA_annots[f]
            
            if f == 0:
                self.create_new_node(frame_number, frame_narrations, frame_STA_annots) #Create a new node
            else:
                #For each new frame, we compare it with all the nodes in the graph so far
                node_scores = self.score_nodes(frame_number)
                node_scores = sorted(node_scores, key=lambda node: -node['score'])
                #And we select the two nodes with the highest similarity score
                top1_node = node_scores[0]
                top2_node = node_scores[1] if len(node_scores)>1 else {'score':0}
                
                #If the network is confident that it is very similar, it merges the frame into the node
                #We want to include as many as possible frames in the same node
                if top1_node['score'] > self.thresh_upper: #and top1_node['score'] - top2_node['score'] > 0.1:
                    self.merge_frame_in_node(top1_node['node_name'], frame_number, frame_narrations, frame_STA_annots) 
                #If the network is confident that it is very dissimilar, it creates a new location
                elif top1_node['score'] < self.thresh_lower:
                    self.create_new_node(frame_number, frame_narrations, frame_STA_annots) #Create a new node
                else:
                    #The frame is ignored if the network is uncertain about the frame
                    continue
            print(self.G)    
        print('-----------The LEN OF THE GRAPH IS SO FAR: ', len(self.G), self.v_id)
        for node in self.G:
            print('The node has: ', len(node['frames_in_node']), node['frames_in_node'], ' frames')
        self.save()

checpoint = '/home/lmur/stillfast_baseline/stillfast/output/sta/StillFast_sta_still_fast_ego4dv2/version_0/checkpoints/epoch=28-step=0237481-map_box_noun_verb_ttc=3.8369.ckpt'
checkpoint = torch.load(checpoint, map_location = 'cpu')['state_dict']
for k, v in checkpoint.items():
    print(k, v.shape)
    
cfg_file = '/home/lmur/stillfast_baseline/stillfast/configs/sta/DINOV2_with_FAST_EK55.yaml'
cfg = get_cfg() # Setup cfg.
cfg.merge_from_file(cfg_file)

dataset = EK55_Environmental_Affordances(split = 'train', cfg = cfg)
print('The length of the dataset is: ', len(dataset))
bar = tqdm.tqdm(total=len(dataset))
#Thing to add: change the dataset in order to add narrations from all the possible narrators.
#Add the load frame function

for n, video_sequence in enumerate(dataset):
    break
    #Check if the graph already exists
    scene_name = video_sequence['video_id'].split('_')[0]
    print(scene_name, int(scene_name[1:]))
    potential_video_path = os.path.join('/home/lmur/catania_lorenzo/EK_data_extracted/output_topo_graphs_TRAIN/local_topological_graphs_COMPLETE', 'topo_graph_' + video_sequence['video_id'] + '.json')
    if os.path.exists(potential_video_path):
        print('YA EXISTE!!, PASAMOS')
        continue

    build_local_graph = Create_Local_Topo_Graph(video_sequence, dataset) 
    build_local_graph.build()
    bar.update()

    
