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

from propagate_AFF_dataset import Ego4Dv1_Propagate_Aff


class Aff_propagator():
    def __init__(self, dataset):
        self.dataset = dataset
        # Train dataset: narration descriptions features
        self.visual_EgoVLP_train_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/videos_EgoVLP'
        self.text_EgoVLP_train_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/narrations_EgoVLP'
        self.train_root = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_TRAIN'
        self.train_nodes = json.load(open('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/clusters/only_nodes.json', 'r'))
        self.still_frames_path = '/home/furnari/data/ego4d/v2-15-02-23/object_frames/'

        # Val dataset: visual descriptions features
        self.val_EgoVLP_feats = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_val_split/videos_EgoVLP'
        #self.val_EgoVLP_feats = '/home/lmur/catania_lorenzo/test_split_data_extracted/videos_EgoVLP_resize224'

        #self.load_training_clusters()
        #self.load_validation_queries()
        self.V_val_desc = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_val_split/V_val_desc.pt', map_location=torch.device('cuda:0'))
        self.V_training_node_mean = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/only_nodes/V_training_desc.pt', map_location=torch.device('cuda:0'))
        self.T_training_node_mean = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/only_nodes/T_training_desc.pt', map_location=torch.device('cuda:0'))
        

    def load_training_clusters(self):
        print('--------Loading training EgoVLP feats----------------')
        self.V_training_node_mean = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)
        self.T_training_node_mean = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)

        total_frames = 0
        for node in tqdm.tqdm(self.train_nodes):
            V_node, T_node = [], []
            local_node = node[0]
            for f in local_node['visits']:
                V_node.append(self.load_single_frame_EgoVLP(self.visual_EgoVLP_train_feats, local_node['v_id'], f))
                T_node.append(self.load_single_frame_EgoVLP(self.text_EgoVLP_train_feats, local_node['v_id'], f))
            total_frames += len(local_node['visits'])
            self.V_training_node_mean.append(torch.mean(torch.cat(V_node, dim=0), dim=0, keepdim=True))
            self.T_training_node_mean.append(torch.mean(torch.cat(T_node, dim=0), dim=0, keepdim=True))
        print('There are', total_frames, 'frames in total, clustered in ', len(self.train_nodes))
        self.V_training_node_mean = torch.cat(self.V_training_node_mean, dim=0)
        self.T_training_node_mean = torch.cat(self.T_training_node_mean, dim=0)

        torch.save(self.V_training_node_mean, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/only_nodes/V_training_desc.pt')
        torch.save(self.T_training_node_mean, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/only_nodes/T_training_desc.pt')
        print('There are', self.V_training_node_mean.shape, 'nodes, with in total')


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
        EgoVLP_video_feats = torch.load(os.path.join(root, f"{v_id}_{f:07d}.pt")).to('cuda:1')
        EgoVLP_video_feats /= EgoVLP_video_feats.norm(dim=-1, keepdim=True)
        return EgoVLP_video_feats

    ## This creates a uniform distribution of the aff inside a node
    def from_STA_node_to_AFF_vector(self, STA_list):
        AFF_nouns, AFF_verbs = np.zeros(self.dataset.n_nouns), np.zeros(self.dataset.n_verbs)
        #AFF_nouns, AFF_verbs = np.ones(self.dataset.n_nouns), np.ones(self.dataset.n_verbs)
        #AFF_nouns /= np.sum(AFF_nouns) #Normalize the distribution
        #AFF_verbs /= np.sum(AFF_verbs) #Normalize the distribution
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

            if len(noun_id) > 1:
                print('que puta mierda pasa')
            
            AFF_verbs[verb_id] = 1
            AFF_nouns[noun_id] = 1
            
        #Add the offset of the background, which is a new component
        AFF_nouns = np.concatenate((np.zeros(1), AFF_nouns))
        AFF_verbs = np.concatenate((np.zeros(1), AFF_verbs))
        

        return AFF_nouns, AFF_verbs

    def visualize_closest_nodes(self, closest_V, closest_T, v_id, frame, sta_noun, sta_verb, AFF_nouns, AFF_verbs):
        #Aff plot graphs
        AFF_N_idx = np.argsort(AFF_nouns)[-10:] - 1 #Remove the background
        AFF_V_idx = np.argsort(AFF_verbs)[-10:] - 1
        AFF_N_label = [val_dataset.nouns_dict[i]['name'] for i in AFF_N_idx]
        AFF_V_label = [val_dataset.verbs_dict[i]['name'] for i in AFF_V_idx]
        AFF_N_probs = AFF_nouns[AFF_N_idx + 1]
        AFF_V_probs = AFF_verbs[AFF_V_idx + 1]

        #Show in a graph this two distributions
        fig, axs = plt.subplots(2, 1)
        barras_n = axs[0].bar(AFF_N_label, AFF_N_probs)
        axs[0].set_xticklabels(AFF_N_label, rotation=45)
        if (sta_noun[0]) in AFF_N_idx:
            gt_idx = np.where(AFF_N_idx == (sta_noun[0]))[0][0]
            barras_n[gt_idx].set_color('orange')

        barras_v = axs[1].bar(AFF_V_label, AFF_V_probs)
        axs[1].set_xticklabels(AFF_V_label, rotation=45)
        if (sta_verb[0]) in AFF_V_idx:
            gt_idx = np.where(AFF_V_idx == (sta_verb[0]))[0][0]
            barras_v[gt_idx].set_color('orange')
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        graph = Image.open(buf)

        #Query image
        query_img = Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{v_id}_{frame:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256))

        #Cluster images
        all_visits, all_text = [], []
        for f, sta_annot in zip(closest_V['visits'], closest_V['STA']):
            all_visits.append((closest_V['v_id'], f))
            all_text.append(sta_annot)

        visits = np.random.choice(len(all_visits), 4, replace=True)
        all_text = [all_text[i] for i in visits]
        cluster_imgs = [Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{all_visits[i][0]}_{all_visits[i][1]:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256)) for i in visits]
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.025, hspace=0.20)
        for i, img in enumerate(cluster_imgs):
            ax = plt.subplot(gs[i])
            ax.set_title(all_text[i], fontsize=8)
            plt.imshow(img)
            plt.axis('off')
        #Save in order to put latter in the joined image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        cluster_grid_V = Image.open(buf)

        #Cluster images
        all_visits, all_text = [], []
        for f, sta_annot in zip(closest_T['visits'], closest_T['STA']):
            all_visits.append((closest_T['v_id'], f))
            all_text.append(sta_annot)
        visits = np.random.choice(len(all_visits), 4, replace=True)
        all_text = [all_text[i] for i in visits]
        cluster_imgs = [Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{all_visits[i][0]}_{all_visits[i][1]:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256)) for i in visits]
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.025, hspace=0.20)
        for i, img in enumerate(cluster_imgs):
            ax = plt.subplot(gs[i])
            ax.set_title(all_text[i], fontsize=8)
            plt.imshow(img)
            plt.axis('off')
        #Save in order to put latter in the joined image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        cluster_grid_T = Image.open(buf)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(query_img)
        plt.axis('off')
        #Save in order to put latter in the joined image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        query_img_show = Image.open(buf)
        
        #Plot three different images in row
        fig, axs = plt.subplots(2, 2, figsize=(14, 14))
        STA_N_label = val_dataset.nouns_dict[sta_noun[0]]['name']
        STA_V_label = val_dataset.verbs_dict[sta_verb[0]]['name']
        axs[0, 0].imshow(query_img_show)
        axs[0, 0].set_title('Query image, STA LABELS ' + STA_V_label + ' ' + STA_N_label)
        axs[0, 0].axis('off')
        axs[0, 1].imshow(graph)
        axs[0, 1].set_title('Predicted AFF')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(cluster_grid_V)
        axs[1, 0].set_title('Frames of the closest VISUAL cluster')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(cluster_grid_T)
        axs[1, 1].set_title('Frames of the closest TEXT cluster')
        axs[1, 1].axis('off')
        plt.tight_layout()
        plt.show()
        #Save the figure
        out_dir = '/home/lmur/catania_lorenzo/data_extracted/AFF_final_Ego4D'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f"{v_id}_{frame:07d}.jpg"), dpi = 300)
        plt.close()

val_dataset = Ego4Dv1_Propagate_Aff()
extrapolate_AFF = Aff_propagator(dataset = val_dataset)
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

N_acc = {'top1': 0, 'top2': 0, 'top3': 0, 'top4': 0, 'top6': 0, 'top8': 0}
V_acc = {'top1': 0, 'top2': 0, 'top3': 0, 'top4': 0, 'top6': 0, 'top8': 0}
number_AFF_N = {'top1': 0, 'top2': 0, 'top3': 0, 'top4': 0, 'top6': 0, 'top8': 0}
number_AFF_V = {'top1': 0, 'top2': 0, 'top3': 0, 'top4': 0, 'top6': 0, 'top8': 0}
STA_prob_N = {'top1': [], 'top2': [], 'top3': [], 'top4': [], 'top6': [], 'top8': []}
STA_prob_V = {'top1': [], 'top2': [], 'top3': [], 'top4': [], 'top6': [], 'top8': []}
n_samples = 1

predicted_AFF = []
softmax = torch.nn.Softmax(dim=0)


for f, frame in enumerate(tqdm.tqdm(val_dataset)):
    frame_desc = extrapolate_AFF.V_val_desc[f, :].unsqueeze(0) #extrapolate_AFF.load_single_frame_EgoVLP(extrapolate_AFF.val_EgoVLP_feats, frame['v_id'], frame['frame'])
    visual_dist_nodes = (1 - cosine_similarity(frame_desc, extrapolate_AFF.V_training_node_mean)).cpu().numpy()
    cross_dist_nodes = (1 - cosine_similarity(frame_desc, extrapolate_AFF.T_training_node_mean)).cpu().numpy()

    #Select the cluster where there is the frame with the lowest distance
    output = {'v_id': frame['v_id'], 'frame': frame['frame']}

    for k in N_acc.keys():
        topk_visual = np.argsort(visual_dist_nodes)[:int(k[3:])]
        topk_cross = np.argsort(cross_dist_nodes)[:int(k[3:])]
        visual_dist = visual_dist_nodes[topk_visual]
        cross_dist = cross_dist_nodes[topk_cross]
        
        if k != 'top1':
            min_V = np.min(visual_dist)
            min_T = np.min(cross_dist)
            #Remove the elements which are the double as the distance
            topk_visual = topk_visual[visual_dist < 1.5 * min_V]
            topk_cross = topk_cross[cross_dist < 1.5 * min_T]
            visual_dist = visual_dist[visual_dist < 1.5 * min_V]
            cross_dist = cross_dist[cross_dist < 1.5 * min_T]

        topk = np.concatenate((topk_visual, topk_cross)) #The closest in both modalities
        #topk = topk_cross
        topk_nodes = [extrapolate_AFF.train_nodes[i][0] for i in topk]

        V_nodes_descriptors_list, T_nodes_descriptors_list = [], []
        
        nodes_AFF_N_list, nodes_AFF_V_list = [], []
        for node in topk_nodes:
            AFF_N_in_node, AFF_V_in_node = extrapolate_AFF.from_STA_node_to_AFF_vector(node['STA'])
            nodes_AFF_N_list.append(AFF_N_in_node)
            nodes_AFF_V_list.append(AFF_V_in_node)

        for v_node in topk_visual:
            V_nodes_descriptors_list.append(extrapolate_AFF.V_training_node_mean[v_node])
        V_nodes_descriptors = torch.stack(V_nodes_descriptors_list, dim=0)
        for t_node in topk_cross:
            T_nodes_descriptors_list.append(extrapolate_AFF.T_training_node_mean[t_node])
        T_nodes_descriptors = torch.stack(T_nodes_descriptors_list, dim=0)
        
        #Weighted mean across the clusters. Closer noodes have more weight
        joined_dist = torch.cat((cosine_similarity(frame_desc, V_nodes_descriptors), cosine_similarity(frame_desc, T_nodes_descriptors)), dim=0)
        #joined_dist = cosine_similarity(frame_desc, T_nodes_descriptors)
        intra_cluster_weight = softmax(joined_dist * 2).cpu().numpy()
        
        for i, w_i in enumerate(intra_cluster_weight):
            nodes_AFF_N_list[i] *= w_i
            nodes_AFF_V_list[i] *= w_i
        AFF_nouns = np.max(nodes_AFF_N_list, axis = 0) #np.sum(nodes_AFF_N_list, axis=0)
        AFF_verbs = np.max(nodes_AFF_V_list, axis = 0)#np.sum(nodes_AFF_V_list, axis=0)

        AFF_nouns /= np.max(AFF_nouns) #Normalize the distribution
        AFF_verbs /= np.max(AFF_verbs)

        AFF_nouns = softmax(torch.tensor(AFF_nouns) * 2).numpy()
        AFF_verbs = softmax(torch.tensor(AFF_verbs) * 2).numpy()
        
        
        STA_prob_N[k].append(AFF_nouns[frame['sta_noun'][0]])
        STA_prob_V[k].append(AFF_verbs[frame['sta_verb'][0]])
        
        unique_N, count_N = np.unique(AFF_nouns, return_counts=True)
        base_N = unique_N[np.argmax(count_N)]
        unique_V, count_V = np.unique(AFF_verbs, return_counts=True)
        base_V = unique_V[np.argmax(count_V)]

        if (frame['sta_verb'][0] + 1) in np.where(AFF_verbs > base_V)[0]:
            V_acc[k] += 1
            
        if (frame['sta_noun'][0] + 1) in np.where(AFF_nouns > base_N)[0]:
            N_acc[k] += 1
        if k == 'top3' and f % 50 == 0:
            extrapolate_AFF.visualize_closest_nodes(topk_nodes[0], topk_nodes[4], frame['v_id'], frame['frame'], 
                                                    frame['sta_noun'], frame['sta_verb'], AFF_nouns, AFF_verbs)
        
        number_AFF_N[k] += len(np.where(AFF_nouns == base_N)[0])
        number_AFF_V[k] += len(np.where(AFF_verbs == base_V)[0])   
        
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
out_dir = '/home/lmur/catania_lorenzo/test_split_data_extracted'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#Save the list as a numpy
#np.save(os.path.join(out_dir, 'AFF_v1_split_T2.npy'), predicted_AFF)


for k in N_acc.keys():
    print('The Accuracy for the top-{} is:'.format(k))
    print(f"Top-{k} nouns: {N_acc[k]/n_samples*100}")
    print(f"Top-{k} verbs: {V_acc[k]/n_samples*100}")
    print('The number of AFF nouns is:', number_AFF_N[k]/n_samples)
    print('The number of AFF verbs is:', number_AFF_V[k]/n_samples)
    print('The mean STA prob nouns is:', np.mean(STA_prob_N[k]))
    print('The mean STA prob verbs is:', np.mean(STA_prob_V[k]))
    print()
    

#1. Crear dataset, que en el get item esten las narraciones y el path de las imágenes




    