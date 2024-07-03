import torch
import numpy as np
import tqdm
import itertools

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import clip
import sys
import os
import json
from transformers import BertTokenizer, BertModel, CLIPTextModel, CLIPTokenizer, CLIPModel

from sklearn.cluster import AgglomerativeClustering
import tqdm
import time
import io

from propagate_AFF_dataset import Ego4D_Propagate_Aff


class Aff_propagator():
    def __init__(self, dataset):
        self.dataset = dataset
        # Train dataset: narration descriptions features
        self.visual_EgoVLP_train_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/videos_EgoVLP'
        self.text_EgoVLP_train_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/narrations_EgoVLP'
        self.train_root = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_TRAIN'
        self.train_clusters = json.load(open('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/clusters/vis_text_1dot50_both.json', 'r'))
        self.still_frames_path = '/home/furnari/data/ego4d/v2-15-02-23/object_frames/'

        # Val dataset: visual descriptions features
        self.val_EgoVLP_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_val_split/videos_EgoVLP'

        #self.load_training_clusters()
        #self.load_validation_queries()
        self.V_val_desc = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_val_split/V_val_desc.pt')
        self.V_training_node_mean = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/V_training_desc.pt')
        self.T_training_node_mean = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/T_training_desc.pt')
        self.nodes_per_cluster = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/nodes_per_cluster.pt')
        

    def load_training_clusters(self):
        print('--------Loading training EgoVLP feats----------------')
        self.V_training_node_mean = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)
        self.T_training_node_mean = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)
        #self.frames_idx = [] #List of C elements, each is the idx of the frames in the cluster
        self.nodes_per_cluster = []
        #node_id = 0
        print('There are', len(self.train_clusters), 'clusters')
        for cluster in tqdm.tqdm(self.train_clusters):
            n_nodes = 0
            for node in cluster:
                V_node, T_node = [], []
                for f in node['visits']:
                    V_node.append(self.load_single_frame_EgoVLP(self.visual_EgoVLP_train_feats, node['v_id'], f))
                    T_node.append(self.load_single_frame_EgoVLP(self.text_EgoVLP_train_feats, node['v_id'], f))
                    #self.frames_idx.append((node['v_id'], f, node_id))
                self.V_training_node_mean.append(torch.mean(torch.cat(V_node, dim=0), dim=0, keepdim=True))
                self.T_training_node_mean.append(torch.mean(torch.cat(T_node, dim=0), dim=0, keepdim=True))
                n_nodes += 1
            self.nodes_per_cluster.append(n_nodes)
        self.V_training_node_mean = torch.cat(self.V_training_node_mean, dim=0)
        self.T_training_node_mean = torch.cat(self.T_training_node_mean, dim=0)
        self.nodes_per_cluster = np.array(self.nodes_per_cluster)
        torch.save(self.V_training_node_mean, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/V_training_desc.pt')
        torch.save(self.T_training_node_mean, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/T_training_desc.pt')
        torch.save(self.nodes_per_cluster, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean_node/nodes_per_cluster.pt')
        print('There are', self.V_training_node_mean.shape, 'clusters, with in total')
        #print(self.frames_idx[0:10])

    def load_validation_queries(self):
        print('--------Loading validation EgoVLP feats----------------')
        self.V_val_desc = []
        for f in tqdm.tqdm(self.dataset):
            v_desc = self.load_single_frame_EgoVLP(self.val_EgoVLP_feats, f['v_id'], f['frame'])
            self.V_val_desc.append(v_desc)
        self.V_val_desc = torch.cat(self.V_val_desc, dim=0)
        print('There are', self.V_val_desc.shape, 'query frames')

    ## -------------------------------Read the CLIP Descriptors of each cluster--------------------------------#
    def load_single_frame_EgoVLP(self, root, v_id, f):
        EgoVLP_video_feats = torch.load(os.path.join(root, f"{v_id}_{f:07d}.pt")).cuda()
        EgoVLP_video_feats /= EgoVLP_video_feats.norm(dim=-1, keepdim=True)
        return EgoVLP_video_feats

    ## This creates a uniform distribution of the aff inside a node
    def from_STA_node_to_AFF_vector(self, STA_list):
        #AFF_nouns, AFF_verbs = np.zeros(self.dataset.n_nouns), np.zeros(self.dataset.n_verbs)
        AFF_nouns, AFF_verbs = np.ones(self.dataset.n_nouns), np.ones(self.dataset.n_verbs)
        AFF_nouns /= np.sum(AFF_nouns) #Normalize the distribution
        AFF_verbs /= np.sum(AFF_verbs) #Normalize the distribution
        for f, sta in enumerate(STA_list):
            STA_interaction = STA_list[f]

            STA_interaction = STA_interaction.replace('vegetable_fruit', 'vegetable')
            STA_interaction = STA_interaction.replace('playing_cards', 'playing')
            STA_interaction = STA_interaction.replace('tape_measure', 'tape')
            STA_interaction = STA_interaction.replace('rubber_band', 'rubber')
            
            sta_verb = STA_interaction.split('_')[:-1]
            sta_verb = '_'.join(sta_verb)
            sta_noun = STA_interaction.split('_')[-1:][0]

            verb_id = [d['id'] for d in self.dataset.verbs_dict if d['name'] == sta_verb]
            noun_id = [d['id'] for d in self.dataset.nouns_dict if d['name'] == sta_noun]
            
            AFF_verbs[verb_id] = 1
            AFF_nouns[noun_id] = 1
        return AFF_nouns, AFF_verbs

    def visualize_closest_cluster(self, topk_clusters, v_id, frame, sta_noun, sta_verb, AFF_nouns, AFF_verbs):
        #Aff plot graphs
        AFF_N_idx = np.argsort(AFF_nouns)[-20:]
        AFF_V_idx = np.argsort(AFF_verbs)[-20:]
        AFF_N_label = [val_dataset.nouns_dict[i]['name'] for i in AFF_N_idx]
        AFF_V_label = [val_dataset.verbs_dict[i]['name'] for i in AFF_V_idx]
        AFF_N_probs = AFF_nouns[AFF_N_idx]
        AFF_V_probs = AFF_verbs[AFF_V_idx]

        #Show in a graph this two distributions
        fig, axs = plt.subplots(2, 1, figsize=(7, 5))
        axs[0].bar(AFF_N_label, AFF_N_probs)
        axs[0].set_title('Predicted AFF Nouns')
        axs[0].set_xticklabels(AFF_N_label, rotation=45)
        axs[1].bar(AFF_V_label, AFF_V_probs)
        axs[1].set_title('Predicted AFF Verbs')
        axs[1].set_xticklabels(AFF_V_label, rotation=45)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        graph = Image.open(buf)

        #Query image
        query_img = Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{v_id}_{frame:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256))

        #Cluster images
        all_visits = []
        for cluster in topk_clusters:
            for node in cluster:
                for f in node['visits']:
                    all_visits.append((node['v_id'], f))
        visits = np.random.choice(len(all_visits), 9, replace=True)
        cluster_imgs = [Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{all_visits[i][0]}_{all_visits[i][1]:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256)) for i in visits]
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace=0.025, hspace=0.05)
        for i, img in enumerate(cluster_imgs):
            ax = plt.subplot(gs[i])
            plt.imshow(img)
            plt.axis('off')
        #Save in order to put latter in the joined image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        cluster_grid = Image.open(buf)
        
        #Plot three different images in row
        fig, axs = plt.subplots(1, 3, figsize=(17, 7))
        STA_N_label = val_dataset.nouns_dict[sta_noun[0]]['name']
        STA_V_label = val_dataset.verbs_dict[sta_verb[0]]['name']
        axs[0].imshow(query_img)
        axs[0].set_title('Query image, STA LABELS ' + STA_V_label + ' ' + STA_N_label)
        axs[0].axis('off')
        axs[1].imshow(graph)
        axs[1].set_title('Predicted AFF')
        axs[1].axis('off')
        axs[2].imshow(cluster_grid)
        axs[2].set_title('Frames of the closest cluster')
        axs[2].axis('off')
        plt.tight_layout()
        plt.show()
        #Save the figure
        out_dir = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/qualitative_results'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f"{v_id}_{frame:07d}.jpg"), dpi = 300)
        plt.close()

val_dataset = Ego4D_Propagate_Aff()
extrapolate_AFF = Aff_propagator(dataset = val_dataset)
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


N_acc = {'top1': 0, 'top5': 0, 'top10': 0}
V_acc = {'top1': 0, 'top5': 0, 'top10': 0}
STA_prob_N = {'top1': 0, 'top5': 0, 'top10': 0}
STA_prob_V = {'top1': 0, 'top5': 0, 'top10': 0}
n_samples = 1

predicted_AFF = []
mean_succes = [0]
mean_failure = [0]

softmax = torch.nn.Softmax(dim=0)

n_f = 0
V_training_cluster_mean = []
T_training_cluster_mean = []
for n in extrapolate_AFF.nodes_per_cluster:
    V_training_cluster_mean.append(torch.mean(extrapolate_AFF.V_training_node_mean[n_f:n_f+n], dim=0, keepdim=True))
    T_training_cluster_mean.append(torch.mean(extrapolate_AFF.T_training_node_mean[n_f:n_f+n], dim=0, keepdim=True))
    n_f += n
V_training_cluster_mean = torch.cat(V_training_cluster_mean, dim=0)
T_training_cluster_mean = torch.cat(T_training_cluster_mean, dim=0)

STA_prob_value = []
for f, frame in enumerate(tqdm.tqdm(val_dataset)):
    #print('the target frame is', frame['v_id'], frame['frame'])
    frame_desc = extrapolate_AFF.V_val_desc[f, :].unsqueeze(0) #extrapolate_AFF.load_single_frame_EgoVLP(extrapolate_AFF.val_EgoVLP_feats, frame['v_id'], frame['frame'])
    visual_dist_clusters = 1 - cosine_similarity(frame_desc, V_training_cluster_mean)
    cross_dist_clusters = 1 - cosine_similarity(frame_desc, T_training_cluster_mean)
    
    #Sum of the normalized distances
    total_dist_norm = (softmax(visual_dist_clusters) + softmax(cross_dist_clusters)).cpu().numpy()
    
    #Select the cluster where there is the frame with the lowest distance
    output = {'v_id': frame['v_id'], 'frame': frame['frame']}
    for k in N_acc.keys():
        topk = np.argsort(total_dist_norm)[:int(k[3:])]

        topk_clusters = [extrapolate_AFF.train_clusters[i] for i in topk]
        topk_distances = [total_dist_norm[i] for i in topk]

        V_nodes_descriptors_list, T_nodes_descriptors_list = [], []
        nodes_AFF_N_list, nodes_AFF_V_list = [], []
        for cluster, cluster_id in zip(topk_clusters, topk):
            nodes_cluster = sum(extrapolate_AFF.nodes_per_cluster[:cluster_id])
            V_nodes_descriptors = extrapolate_AFF.V_training_node_mean[nodes_cluster:nodes_cluster+extrapolate_AFF.nodes_per_cluster[cluster_id]]
            V_nodes_descriptors_list.append(V_nodes_descriptors)
            T_nodes_descriptors = extrapolate_AFF.T_training_node_mean[nodes_cluster:nodes_cluster+extrapolate_AFF.nodes_per_cluster[cluster_id]]
            T_nodes_descriptors_list.append(T_nodes_descriptors)
            for node in cluster:
                AFF_N_in_node, AFF_V_in_node = extrapolate_AFF.from_STA_node_to_AFF_vector(node['STA'])
                nodes_AFF_N_list.append(AFF_N_in_node)
                nodes_AFF_V_list.append(AFF_V_in_node)
        V_nodes_descriptors = torch.cat(V_nodes_descriptors_list, dim=0)
        T_nodes_descriptors = torch.cat(T_nodes_descriptors_list, dim=0)
        
        #Weighted mean across the clusters. Closer noodes have more weight
        intra_cluster_visual = softmax(cosine_similarity(frame_desc, V_nodes_descriptors)).cpu().numpy()
        intra_cluster_cross = softmax(cosine_similarity(frame_desc, T_nodes_descriptors)).cpu().numpy()
        intra_cluster_weight = (intra_cluster_visual + intra_cluster_cross) / 2

        #intra_cluster_weight = softmax(cosine_similarity(frame_desc, nodes_descriptors)).cpu().numpy()
        for i, w_i in enumerate(intra_cluster_weight):
            nodes_AFF_N_list[i] *= w_i
            nodes_AFF_V_list[i] *= w_i
        AFF_nouns = np.sum(nodes_AFF_N_list, axis=0)
        AFF_verbs = np.sum(nodes_AFF_V_list, axis=0)
        
        if frame['sta_verb'] in np.where(AFF_verbs > 1 / val_dataset.n_verbs)[0]:
            V_acc[k] += 1
            
        if frame['sta_noun'] in np.where(AFF_nouns > 1 / val_dataset.n_nouns)[0]:
            N_acc[k] += 1
        
        AFF_nouns /= np.sum(AFF_nouns) #Normalize the distribution
        AFF_verbs /= np.sum(AFF_verbs) #Normalize the distribution
        STA_prob_N[k] += AFF_nouns[frame['sta_noun'][0]]
        STA_prob_V[k] += AFF_verbs[frame['sta_verb'][0]]
        
        #if k == 'top10' and f % 50 == 0:
        #    extrapolate_AFF.visualize_closest_cluster(topk_clusters, frame['v_id'], frame['frame'], frame['sta_noun'], frame['sta_verb'], AFF_nouns, AFF_verbs)
        
        output['AFF_N_{}'.format(k)] = AFF_nouns
        output['AFF_V_{}'.format(k)] = AFF_verbs
    
        #print(N_acc[k]/n_samples * 100, V_acc[k]/n_samples * 100, 'Acc score', k)
        #print(STA_prob_N[k]/n_samples, STA_prob_V[k]/n_samples, 'STA prob', k)
        #print('****************')
    predicted_AFF.append(output)      
    n_samples += 1
    #print('The mean success is:', np.mean(mean_succes), 'The mean failure is:', np.mean(mean_failure))
    #print()

#Save the output list
out_dir = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#Save the list as a numpy
np.save(os.path.join(out_dir, 'predicted_AFF_1dot50_weighted.npy'), predicted_AFF)


for k in N_acc.keys():
    print('The Accuracy for the top-{} is:'.format(k))
    print(f"Top-{k} nouns: {N_acc[k]/n_samples*100}")
    print(f"Top-{k} verbs: {V_acc[k]/n_samples*100}")
    print(f"Top-{k} nouns prob: {STA_prob_N[k]/n_samples}")
    print(f"Top-{k} verbs prob: {STA_prob_V[k]/n_samples}")


#1. Crear dataset, que en el get item esten las narraciones y el path de las imágenes




    