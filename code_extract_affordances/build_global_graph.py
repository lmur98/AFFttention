import torch
import numpy as np
import tqdm
import itertools

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/lmur/hum_obj_int/stillfast')
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video

import math
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
from transformers import BertTokenizer, BertModel, CLIPTextModel, CLIPTokenizer, CLIPModel

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import random
import sys
sys.path.append('/home/lmur/hum_obj_int/stillfast')
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video
import time

class Combiner():
    def __init__(self):
        self.root = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_TRAIN'
        self.topo_dir = os.path.join(self.root, 'local_topological_graphs_REVIEW_v2')
        self.EgoVLP_visual_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/videos_EgoVLP'
        self.EgoVLP_text_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/narrations_EgoVLP'
        self.total_vids = self.get_vids_to_combine()
        self.cosine_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)

    def get_vids_to_combine(self):
        vids_to_combine = []
        for file in os.listdir(self.topo_dir):
            if file.endswith('.json'):
                vids_to_combine.append(file.split('_')[-1].split('.')[0])
        return vids_to_combine
    

    def load_CLIP_features(self, v_id, node_name):
        CLIP_dir = os.path.join(self.root, 'CLIP_features_REVIEW_v2')
        CLIP_features = np.load(os.path.join(CLIP_dir, v_id, str(node_name) + '.npy'), allow_pickle=True).item()
        return torch.tensor(CLIP_features['text']).clone().detach(), CLIP_features['visual'].clone().detach()

    def load_single_frame_EgoVLP(self, root, v_id, f):
        EgoVLP_video_feats = torch.load(os.path.join(root, f"{v_id}_{f:07d}.pt")).cuda()
        EgoVLP_video_feats /= EgoVLP_video_feats.norm(dim=-1, keepdim=True)
        f_idx = (v_id, f)
        return EgoVLP_video_feats, f_idx

    def join_all_local_graphs(self):
        all_node_data = [] #List of N dictionaries (total number of all the nodes)
        print('Loading all the local graphs')
        for v_id in tqdm.tqdm(self.total_vids):
            local_graph = json.load(open(os.path.join(self.topo_dir, 'topo_graph_' + v_id + '.json'), 'r'))
            #Local graph is a list of nodes, each of them is a dictionary
            for node in local_graph:
                node_name = node['node_name']
                all_node_data.append({'v_id': v_id, 
                                      'text': node['narrator_1'], #node['STA'],
                                      'node_name': node['node_name'], 
                                      'visits': node['frames_in_node'],
                                      'STA': node['STA'],})
        return all_node_data

    def EgoVLP_distance(self, all_node_data):
        N = len(all_node_data) #N_nodes
        X_v = torch.zeros((N, N))
        X_t = torch.zeros((N, N))
        X_cross = torch.zeros((N, N))
        text_descriptors = []
        video_descriptors = []
        idx_node = []
        print('Loading the EgoVLP descriptions')
        for n in tqdm.tqdm(range(N)):
            video_id = all_node_data[n]['v_id']
            visits = all_node_data[n]['visits']
            for f in visits:
                text_features, _ = self.load_single_frame_EgoVLP(self.EgoVLP_text_feats, video_id, f)
                video_features, _ = self.load_single_frame_EgoVLP(self.EgoVLP_visual_feats, video_id, f)
                text_descriptors.append(text_features)
                video_descriptors.append(video_features)
            idx_node.append(len(visits))
        text_descriptors = torch.cat(text_descriptors, dim = 0) #N_frames, 4096
        video_descriptors = torch.cat(video_descriptors, dim = 0) #N_frames, 4096

        for i, j in tqdm.tqdm(itertools.combinations(range(N), 2), total=(N*(N-1))//2):
            idx_i = sum(idx_node[0:i])
            idx_j = sum(idx_node[0:j])
            
            text_i, visual_i = text_descriptors[idx_i:idx_i + idx_node[i]], video_descriptors[idx_i:idx_i + idx_node[i]]
            text_j, visual_j = text_descriptors[idx_j:idx_j + idx_node[j]], video_descriptors[idx_j:idx_j + idx_node[j]]
            
            #text_dist = self.cosine_sim(text_i.unsqueeze(1), text_j.unsqueeze(0)).mean()
            #visual_dist = self.cosine_sim(visual_i.unsqueeze(0), visual_j.unsqueeze(1)).mean()
            cross_dist = self.cosine_sim(text_i.unsqueeze(1), visual_j.unsqueeze(0)).mean() + self.cosine_sim(visual_i.unsqueeze(1), text_j.unsqueeze(0)).mean()
            #X_t[i, j] = text_dist
            #X_t[j, i] = text_dist
            #X_v[i, j] = visual_dist
            #X_v[j, i] = visual_dist
            X_cross[i, j] = cross_dist
            X_cross[j, i] = cross_dist
        
        #torch.save(X_t, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/sim_matrices/text.pt')
        #torch.save(X_v, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/sim_matrices/visual.pt')
        torch.save(X_cross, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/sim_matrices/cross.pt')

        print(X_v.shape, X_t.shape, 'shape of the descriptors')

    
    def cluster(self):
        X_t = 1 - torch.load(os.path.join('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/sim_matrices', 'text.pt'))
        X_v = 1 - torch.load(os.path.join('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/sim_matrices', 'visual.pt'))
        print('The mean visual dis is', X_v.mean(), 'and text is', X_t.mean())
        print('The max visual dis is', X_v.max(), 'and text is', X_t.max())
        
        #X_t = np.where(X_t > 0.7, 10.0, X_t)
        X_v = np.where(X_v > 0.75, 10.0, X_v)
        X = X_t + X_v
        print('The mean visual dis is', X_v.mean(), 'and text is', X_t.mean())
        clf = AgglomerativeClustering(affinity = 'precomputed', n_clusters = None, distance_threshold = 0, linkage = 'average', compute_full_tree = True).fit(X)
        labels = clf.labels_
        print('From', X_v.shape[0], 'nodes to ', max(labels))
        
        print('The clustered labels are:', labels)
        print(labels.shape)
        print(max(labels))
        clusters = []
        for c_id in set(labels):
            clusters.append([{'v_id': all_node_data[i]['v_id'],
                              'node_name': all_node_data[i]['node_name'], 
                              'visits': all_node_data[i]['visits'],
                              'text': all_node_data[i]['text'],
                              'STA': all_node_data[i]['STA']} for i in np.arange(len(labels))[labels==c_id]]) #'text': all_node_data[i]['text']
        
        #Save clusters
        json.dump(clusters, open(os.path.join('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/clusters', 'only_nodes.json'), 'w'))
        #json.dump(clusters, open(os.path.join('/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_TRAIN/clusters_REVIEW_v2', 'clusters_d030_xv_plus_vt.json'), 'w'))
        #print('We have ', len(clusters), 'clusters extracted from', len(all_node_data), 'nodes')
        return clusters

    def draw_global_graph(self, clusters):
        video_dataset = Ego4DHLMDB_STA_Still_Video('/ssd/furnari/sta_lmdb/', readonly=True, lock=False)
        output_dir = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/vis_clusters/vis_1dot80_mixed'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print('Drawing clusters with some nodes')
        for n, cluster in tqdm.tqdm(enumerate(clusters)):
            v_id_list, frames_list, texts_list = [], [], []
            if len(cluster) < 5:
                continue
            for node in cluster:
                node_v_id = node['v_id']
                node_frames = node['visits']
                node_texts = node['STA']
                for frame, text in zip(node_frames, node_texts):
                    v_id_list.append(node_v_id)
                    frames_list.append(frame)
                    texts_list.append(text)
            #Randomly sample 20 images from the cluster
            num_images = min(len(v_id_list), 20)
            if len(v_id_list) > 20:
                random_choice = np.random.choice(len(v_id_list), 20, replace=False)
                v_id_list = [v_id_list[i] for i in random_choice]
                frames_list = [frames_list[i] for i in random_choice]
                texts_list = [texts_list[i] for i in random_choice]
            columns = 5  # por ejemplo, puedes ajustar esto
            rows = math.ceil(num_images / columns)

            # Configura la figura
            fig, axs = plt.subplots(rows, columns, figsize=(20, rows * 4))  # Ajusta el tamaño según sea necesario
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            if rows == 1 or columns == 1:
                axs = np.array(axs).reshape(rows, columns)

            # Llenar cada subplot
            for i in range(rows):
                for j in range(columns):
                    img_index = i * columns + j
                    if img_index < num_images:
                        img = video_dataset.get(v_id_list[img_index], frames_list[img_index]).convert('RGB').resize((320, 256)).crop((0, 0, 224, 224))
                        axs[i, j].imshow(img)
                        axs[i, j].set_title(texts_list[img_index])
                    axs[i, j].axis('off')  # Oculta los ejes para todos

            fig.suptitle('Cluster: ' + str(n) + ' formed by ' + str(len(cluster)) + ' nodes', fontsize=16)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(output_dir, 'cluster' + str(n) + '.png')) # dpi = 300
            plt.close()
                            

if __name__ == '__main__':
    combiner = Combiner()
    all_node_data = combiner.join_all_local_graphs()
    #combiner.EgoVLP_distance(all_node_data)
    clusters = combiner.cluster()
    #combiner.draw_global_graph(clusters)
