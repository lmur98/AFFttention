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
        self.T_training_desc = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean/T_training_desc.pt')
        self.V_training_desc = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean/V_training_desc.pt')
        #self.frames_idx = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean/frames_idx.pt')
        #self.frames_per_cluser = torch.load('/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot50_mean/frames_per_cluser.pt')
        print(self.T_training_desc.shape, self.V_training_desc.shape)
        

    def load_training_clusters(self):
        print('--------Loading training EgoVLP feats----------------')
        self.V_training_desc = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)
        self.T_training_desc = [] #List of C elemnts, each is a tensor of descriptors of shape (f_clust, 512)
        self.frames_per_cluser = [] #List of C elements, each is the number of frames per cluster
        self.frames_idx = [] #List of C elements, each is the idx of the frames in the cluster
        cluster_id = 0
        for cluster in tqdm.tqdm(self.train_clusters):
            frames_in_cluster = 0
            for node in cluster:
                for f in node['visits']:
                    v_desc = self.load_single_frame_EgoVLP(self.visual_EgoVLP_train_feats, node['v_id'], f)
                    self.V_training_desc.append(v_desc)
                    t_desc = self.load_single_frame_EgoVLP(self.text_EgoVLP_train_feats, node['v_id'], f)
                    self.T_training_desc.append(t_desc)
                    f_idx = (node['v_id'], f, cluster_id)
                    self.frames_idx.append(f_idx)
                    frames_in_cluster += 1
            self.frames_per_cluser.append(frames_in_cluster)
            cluster_id += 1
        self.V_training_desc = torch.cat(self.V_training_desc, dim=0)
        self.T_training_desc = torch.cat(self.T_training_desc, dim=0)
        torch.save(self.V_training_desc, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot60/V_training_desc.pt')
        torch.save(self.T_training_desc, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot60/T_training_desc.pt')
        torch.save(self.frames_idx, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot60/frames_idx.pt')
        torch.save(self.frames_per_cluser, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_train_split/1dot60/frames_per_cluser.pt')
        print('There are', self.V_training_desc.shape, 'clusters, with in total', sum(self.frames_per_cluser), 'frames')
        print(self.frames_idx[0:10])

    def load_validation_queries(self):
        print('--------Loading validation EgoVLP feats----------------')
        self.V_val_desc = []
        for f in tqdm.tqdm(self.dataset):
            v_desc = self.load_single_frame_EgoVLP(self.val_EgoVLP_feats, f['v_id'], f['frame'])
            self.V_val_desc.append(v_desc)
        self.V_val_desc = torch.cat(self.V_val_desc, dim=0)
        torch.save(self.V_val_desc, '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/STA_val_split/V_val_desc.pt')
        print('There are', self.V_val_desc.shape, 'query frames')

    ## -------------------------------Read the CLIP Descriptors of each cluster--------------------------------#
    def load_single_frame_EgoVLP(self, root, v_id, f):
        EgoVLP_video_feats = torch.load(os.path.join(root, f"{v_id}_{f:07d}.pt")).cuda()
        EgoVLP_video_feats /= EgoVLP_video_feats.norm(dim=-1, keepdim=True)
        return EgoVLP_video_feats

    def STA_to_AFF(self, AFF_nouns, AFF_verbs, STA_list):
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
            
            AFF_verbs[verb_id] += 1
            AFF_nouns[noun_id] += 1
        return AFF_nouns, AFF_verbs

    def visualize_closest_cluster(self, topk_clusters, v_id, frame, sta):
        print('The top-1 cluster is:')
        query_img = Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{v_id}_{frame:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256))

        all_visits = []
        AFF_nouns = np.zeros(val_dataset.n_nouns)
        AFF_verbs = np.zeros(val_dataset.n_verbs) 
        for node in topk_clusters[0]:
            for f in node['visits']:
                all_visits.append((node['v_id'], f))
            AFF_nouns, AFF_verbs = extrapolate_AFF.STA_to_AFF(AFF_nouns, AFF_verbs, node['STA'])
        #Select 4 visits
        visits = np.random.choice(len(all_visits), 4, replace=True)
        cluster_imgs = [Image.open(os.path.join(extrapolate_AFF.still_frames_path, f"{all_visits[i][0]}_{all_visits[i][1]:07d}.jpg")).resize((320, 256)).crop((0, 0, 256, 256)) for i in visits]
        AFF_N_idx = np.where(AFF_nouns > 0)[0]
        AFF_V_idx = np.where(AFF_verbs > 0)[0]
        AFF_N_label = [val_dataset.nouns_dict[i]['name'] for i in AFF_N_idx]
        AFF_V_label = [val_dataset.verbs_dict[i]['name'] for i in AFF_V_idx]
            
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].imshow(query_img)
        axs[0, 0].set_title('Query image, STA LABELS ' + sta)
        axs[0, 0].axis('off')
        axs[1, 0].axis('off')

        nouns_color = [val_dataset.cmap_n[i] for i in AFF_N_idx]
        verbs_color = [val_dataset.cmap_v[i] for i in AFF_V_idx]

        nouns_line = [plt.Line2D([0,0],[0,0],color=c, marker='o', linestyle='') for c in nouns_color]
        verbs_line = [plt.Line2D([0,0],[0,0],color=c, marker='o', linestyle='') for c in verbs_color]

        nouns_legend = fig.legend(nouns_line, AFF_N_label, loc='lower left', title='Nouns', fontsize='large')
        verbs_legend = fig.legend(verbs_line, AFF_V_label, loc='lower right', title='Verbs', fontsize='large')

        plt.gca().add_artist(nouns_legend)
        plt.gca().add_artist(verbs_legend)

        for i, img in enumerate(cluster_imgs):
            axs[i//2, i%2 + 1].imshow(img)
            axs[i//2, i%2 + 1].axis('off')
        plt.suptitle('Frames of the closest cluster')
            
        plt.show()
        #Save the figure
        out_dir = '/home/lmur/catania_lorenzo/data_extracted/data_EgoVLP/results'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f"{v_id}_{frame:07d}.jpg"))
        plt.close()

val_dataset = Ego4D_Propagate_Aff()
extrapolate_AFF = Aff_propagator(dataset = val_dataset)
cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


N_acc = {'top1': 0, 'top3': 0, 'top5': 0}
V_acc = {'top1': 0, 'top3': 0, 'top5': 0}
mean_AFF_N = {'top1': 0, 'top3': 0, 'top5': 0}
mean_AFF_V = {'top1': 0, 'top3': 0, 'top5': 0}
n_samples = 1

predicted_AFF = []
mean_succes = [0]
mean_failure = [0]
    
for f, frame in enumerate(tqdm.tqdm(val_dataset)):
    #print('the target frame is', frame['v_id'], frame['frame'])
    frame_desc = extrapolate_AFF.V_val_desc[f, :].unsqueeze(0) #extrapolate_AFF.load_single_frame_EgoVLP(extrapolate_AFF.val_EgoVLP_feats, frame['v_id'], frame['frame'])
    visual_dist_clusters = 1 - cosine_similarity(frame_desc, extrapolate_AFF.V_training_desc).cpu().numpy()
    cross_dist_clusters = 1 - cosine_similarity(frame_desc, extrapolate_AFF.T_training_desc).cpu().numpy()
    
    #Select the cluster where there is the frame with the lowest distance
    
    #visual_dist_clusters, cross_dist_clusters = [], []
    #n_f = 0
    #for n in extrapolate_AFF.frames_per_cluser:
    #    visual_dist_clusters.append(np.mean(visual_dist_frames[n_f:n_f+n]))
    #    cross_dist_clusters.append(np.mean(cross_dist_frames[n_f:n_f+n]))
    #    n_f += n
    #visual_dist_clusters = np.array(visual_dist_clusters)
    #cross_dist_clusters = np.array(cross_dist_clusters)

    output = {'v_id': frame['v_id'], 'frame': frame['frame']}
    for k in N_acc.keys():
        #topk = np.argsort(visual_dist_clusters) #[:int(k[-1])]
        #topk = np.argsort(visual_dist_frames)[:int(k[-1])]
        #topk_frames = [extrapolate_AFF.frames_idx[i] for i in topk] #[:int(k[-1])]
        #topk_clusters = [extrapolate_AFF.train_clusters[i[2]] for i in topk_frames]
        #topk_distances = [visual_dist_frames[i] for i in topk]
        topk_idx = np.argsort(visual_dist_clusters)[:int(k[-1])]
        topk_clusters = [extrapolate_AFF.train_clusters[i] for i in topk_idx]
        topk_distances = [visual_dist_clusters[i] for i in topk_idx]
        
        #topk_frames = [i for i, j in itertools.groupby(topk_frames, lambda x: x[2])]
        #topk = np.argsort(visual_dist_clusters)[:int(k[-1])]
        #Select the elements with a distance less than 0.7. Otherwise, select only the first

        #if topk != 'top1':
        #    topk = topk[visual_dist_clusters[topk] < 0.75]
        #if len(topk) == 0:  
        #    topk = np.argsort(visual_dist_clusters)[:1]
        #topk_clusters = [extrapolate_AFF.train_clusters[i] for i in topk]

        AFF_nouns = np.zeros(val_dataset.n_nouns)
        AFF_verbs = np.zeros(val_dataset.n_verbs)
        for cluster in topk_clusters:
            for node in cluster:
                AFF_nouns, AFF_verbs = extrapolate_AFF.STA_to_AFF(AFF_nouns, AFF_verbs, node['STA'])
        
        mean_AFF_N[k] += len(np.where(AFF_nouns > 0)[0])
        mean_AFF_V[k] += len(np.where(AFF_verbs > 0)[0])

        if frame['sta_verb'] in np.where(AFF_verbs > 0)[0]:
            V_acc[k] += 1
            
        if frame['sta_noun'] in np.where(AFF_nouns > 0)[0]:
            N_acc[k] += 1
            #print('Success')
            if k == 'top5':
                mean_succes.append(np.mean(topk_distances))
        else:
            #print('Failure')
            if k == 'top3':
                if topk_distances[0] < 0.4:
                    print('FALLO GORDO', topk_distances)
            if k == 'top5':
                mean_failure.append(np.mean(topk_distances))
        #print('The visual distances are:', visual_dist_clusters[topk_idx])
        #print('The cross distances are:', cross_dist_clusters[topk_idx])

        
        #else:
        #    print('wrooong, we show it')
        #    extrapolate_AFF.visualize_closest_cluster(topk_clusters, frame['v_id'], frame['frame'], str(frame['sta_noun']) + ' ' + str(frame['sta_verb']))
        

        output['AFF_N_{}'.format(k)] = np.where(AFF_nouns > 0)[0]
        output['AFF_V_{}'.format(k)] = np.where(AFF_verbs > 0)[0]
        #output['Temperature_{}'.format(k)] = np.mean(visual_dist_frames[topk])
    
        #print(N_acc[k]/n_samples * 100, V_acc[k]/n_samples * 100, 'Acc score')
        #print(mean_AFF_N[k]/n_samples, mean_AFF_V[k]/n_samples, 'Mean number of AFF')
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
np.save(os.path.join(out_dir, 'predicted_AFF_1dot50.npy'), predicted_AFF)


for k in N_acc.keys():
    print('The Accuracy for the top-{} is:'.format(k))
    print(f"Top-{k} nouns: {N_acc[k]/n_samples*100}")
    print(f"Top-{k} verbs: {V_acc[k]/n_samples*100}")
    print(mean_AFF_N[k]/n_samples, mean_AFF_V[k]/n_samples, 'Mean number of AFF')
print('The mean success is:', np.mean(mean_succes), 'The mean failure is:', np.mean(mean_failure))

#1. Crear dataset, que en el get item esten las narraciones y el path de las imágenes




    