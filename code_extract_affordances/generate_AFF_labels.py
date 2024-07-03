#This files reads the cluster.json and generates the affordances labels for each image in the STA dataset-----------------
#AFF_EGO4D_V1.json has the following structure:
#List of Dict  {'v_id': str, 'frame': int, 'aff_verbs': np.array(), 'aff_nouns': np.array(), 'STA': str}
#aff_verbs and aff_nouns are arrays of strings, each contains the probability of the verb/noun to be present in the image
#STA is the text of the STA

import os
import json
import numpy as np
import tqdm

nouns_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_nouns_list.json', 'r'))
adapted_nouns_dict = []
for noun in nouns_dict:
    new_noun = {'id': noun['id'], 'name': noun['name'].split('_')[0]}
    adapted_nouns_dict.append(new_noun)

verbs_dict = json.load(open('/home/lmur/catania_lorenzo/code/extract_affordances/STA_verbs_list.json', 'r'))
adapted_verbs_dict = []
for verb in verbs_dict:
    new_verb = {'id': verb['id'], 'name': verb['name'].split('_(')[0]}
    adapted_verbs_dict.append(new_verb)

root = '/home/lmur/catania_lorenzo/data_extracted/output_topo_graphs_TRAIN'
clusters_json_path = os.path.join(root, 'clusters_REVIEW', 'clusters_d045_complete.json')
clusters_json = json.load(open(clusters_json_path, 'r'))
print('the len of the clusters are', len(clusters_json))
AFF = []
for cluster in tqdm.tqdm(clusters_json):
    AFF_verbs = np.zeros(len(verbs_dict))
    AFF_nouns = np.zeros(len(nouns_dict))
    n_samples = 0
    all_visits = []
    for node_clustered in cluster:
        v_id = node_clustered['v_id']
        visits = node_clustered['visits']
        STA = node_clustered['STA']
        for v, visit in enumerate(visits):
            all_visits.append((v_id, visit))

        for f, sta in enumerate(STA):
            STA_interaction = STA[f]
            STA_interaction = STA_interaction.replace('vegetable_fruit', 'vegetable')
            STA_interaction = STA_interaction.replace('playing_cards', 'playing')
            STA_interaction = STA_interaction.replace('tape_measure', 'tape')
            STA_interaction = STA_interaction.replace('rubber_band', 'rubber')
            
            sta_verb = STA_interaction.split('_')[:-1]
            sta_verb = '_'.join(sta_verb)
            sta_noun = STA_interaction.split('_')[-1:][0]

            verb_id = [d['id'] for d in adapted_verbs_dict if d['name'] == sta_verb]
            noun_id = [d['id'] for d in adapted_nouns_dict if d['name'] == sta_noun]
            
            AFF_verbs[verb_id] += 1
            AFF_nouns[noun_id] += 1
            n_samples += 1

    AFF_verbs /= n_samples
    AFF_nouns /= n_samples
    
    
    for v, visit in enumerate(all_visits):
        #For each visit, we select all the visits that share the same STA
        v_id_n = all_visits[v][0]
        frame_n = all_visits[v][1]
        #visits_with_same_STA = []
        #for f, visit in enumerate(all_visits):
        #    v_id_f = all_visits[f][0]
        #    frame_f = all_visits[f][1]    
        #    STA_f = all_visits[f][2]
        #    if STA_n == STA_f and frame_f != frame_n:
        #        visits_with_same_STA.append((v_id_f, frame_f))

        #All visits inside the cluster share the same affordances
        #visits_with_same_AFF = []
        #for f, visit in enumerate(all_visits):
        #    v_id_f = all_visits[f][0]
        #    frame_f = all_visits[f][1]    
        #    if frame_f != frame_n:
        #        visits_with_same_AFF.append((v_id_f, frame_f))

        AFF.append({'v_id': v_id_n, 
                    'frame': frame_n, 
                    #'visits_with_STA': visits_with_same_STA,
                    #'visits_with_AFF': visits_with_same_AFF,
                    'aff_verbs': AFF_verbs.tolist(), 
                    'aff_nouns': AFF_nouns.tolist()})

with open(os.path.join(root, 'EGO4D_AFF_REVIEW', 'AFF_EGO4D_V1_VAL_complete_d045.json'), 'w') as f:
    json.dump(AFF, f)


        