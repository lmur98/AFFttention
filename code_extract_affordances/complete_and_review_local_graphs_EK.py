import json
import sys
sys.path.append('/home/lmur/stillfast_baseline/stillfast')
from stillfast.config.defaults import get_cfg #Import the configurations
from ek55_env_aff_dataset import EK55_Environmental_Affordances
import os
import tqdm
import numpy as np

class Complete_Local_Graphs():
    def __init__(self, root):
        self.local_videos_dir = os.path.join(root, 'local_topological_graphs_COMPLETE')
        self.output_dir = os.path.join(root, 'local_topological_graphs_REVIEWED')
    
    def read_local_graph(self, v_id):
        local_graph = json.load(open(os.path.join(self.local_videos_dir, 'topo_graph_' + v_id + '.json'), 'r'))
        return local_graph

    def from_STA_to_narrations(self, STA_annots):
        narrations = []
        for f, STA in enumerate(STA_annots):
            STA_narration = '#C C ' + STA['gt_verb_names'][0] + ' ' + STA['gt_noun_names'][0]
            narrations.append({'frame_number': STA['frame_number'],
                               'begin_narration': None,
                               'end_narration': None,
                               'narrator_1': STA_narration})
        return narrations

    def identify_missing_frames(self, video_sequence, local_graph):
        # Convert 'frames_in_node' for each node in local_graph into a set for faster lookup
        node_frame_sets = [set(node['frames_in_node']) for node in local_graph]

        # Use a set comprehension to find missing frames efficiently
        missing_frames = {
            frame for frame in video_sequence['frames_list']
            if not any(frame in node_frames for node_frames in node_frame_sets)
        }
        print('Missing frames: ', missing_frames)
        return list(missing_frames)

    def complete_with_missing(self, video_sequence, local_graph, missing_frames):
        for key in video_sequence:
            print(key, len(video_sequence[key]))
        for frame in missing_frames:
            new_node = self.create_new_node(frame, video_sequence)
            local_graph.append(new_node)
        return local_graph

    def create_new_node(self, missed_frame, video_sequence):
        missed_frame_idx = video_sequence['frames_list'].index(missed_frame)
        
        for key in video_sequence:
            print(key, len(video_sequence[key]))
            
        missed_narration = video_sequence['narrations'][missed_frame_idx]
        missed_STA = video_sequence['targets'][missed_frame_idx]
        n_annots = len(missed_STA['verb_category_id'])
        missed_STA_verb = []
        missed_STA_noun = []
        for n in range(n_annots):
            missed_STA_verb.append(missed_STA['verb_category_id'][n])
            missed_STA_noun.append(missed_STA['noun_category_id'][n])            

        new_node = {'node_name': missed_frame, 
                    'frames_in_node': [missed_frame], 
                    'narrator': missed_narration, 
                    'STA_noun': missed_STA_noun,
                    'STA_verb': missed_STA_verb,}
        return new_node

    def save(self, new_local_graph, video_id):
        with open(os.path.join(self.output_dir, 'topo_graph_' + video_id + '.json'), 'w') as f:
            json.dump(new_local_graph, f)

class Revise_Topo_Graph():
    def __init__(self):
        self.root = '/home/lmur/catania_lorenzo/v2_data_extracted/output_topo_graphs_TRAIN'
        self.input_dir = os.path.join(self.root, 'local_topological_graphs_COMPLETE')
        
    def analyze_graph(self, graph):
        self.graph = graph
        #Add to the temporal closest node
        for Q_node in self.graph:
            frames_in_node = Q_node['frames_in_node']
            if len(frames_in_node) == 1:
                self.add_to_closest_T(Q_node) #Q_node: query node, which we want to join
        
        #Remove nodes with only one frame
        self.out_G = []
        for Q_node in self.graph:
            frames_in_node = Q_node['frames_in_node']
            if len(frames_in_node) > 1:
                self.out_G.append(Q_node)
            else:
                single_frame = frames_in_node[0]
                aparitions = 0
                #Check that the frame is not in any other node, except itself
                for R_node in self.graph:
                    if single_frame in R_node['frames_in_node']:
                        aparitions += 1
                if aparitions == 1:
                    print('Dont remove, it is alone!')
                    self.out_G.append(Q_node)

        return self.out_G        
    
    def add_to_closest_T(self, Q_node):
        T_min = 10000
        Node_T_min = None
        for R_node in self.graph:
            if len(R_node['frames_in_node']) > 1:
                T_dist = self.compute_T_dist(Q_node, R_node)
                if T_dist < T_min:
                    T_min = T_dist
                    Node_T_min = R_node
        if Node_T_min is not None:
            self.join_nodes(Q_node, Node_T_min)
        
    def compute_T_dist(self, Q_node, R_node):
        Q_node = np.asarray(Q_node['frames_in_node'])
        minimun = (np.abs(R_node['frames_in_node'] - Q_node)).min()
        return minimun

    def join_nodes(self, Q_node, R_node):
        for node in self.graph:
            if node['node_name'] == R_node['node_name']:
                node['frames_in_node'].append(Q_node['frames_in_node'][0])
                if len(Q_node['narrator']) > 0:
                    node['narrator'].append(Q_node['narrator'][0])
                node['STA_noun'].append(Q_node['STA_noun'][0])
                node['STA_verb'].append(Q_node['STA_verb'][0])
                break
   

cfg_file = '/home/lmur/stillfast_baseline/stillfast/configs/sta/STILL_FAST_R50_X3DM_EK55.yaml'
cfg = get_cfg() # Setup cfg.
cfg.merge_from_file(cfg_file)
dataset = EK55_Environmental_Affordances(split = 'train', cfg = cfg)

complete_local_graphs = Complete_Local_Graphs(root = '/home/lmur/catania_lorenzo/EK_data_extracted/output_topo_graphs_TRAIN')
review_local_graphs = Revise_Topo_Graph()

for n, video_sequence in tqdm.tqdm(enumerate(dataset)):
    video_id = video_sequence['video_id']
    local_graph = complete_local_graphs.read_local_graph(video_id) #Load the graph with the video_id name
    missing_frames = complete_local_graphs.identify_missing_frames(video_sequence, local_graph) #Identify the missing frames
    print('THE MISSING FRAMES ARE', missing_frames)
    new_local_graph = complete_local_graphs.complete_with_missing(video_sequence, local_graph, missing_frames) #Complete the graph with the missing frames
    revised_local_graph = review_local_graphs.analyze_graph(new_local_graph) #Revise the graph
    complete_local_graphs.save(revised_local_graph, video_id) #Save the new graph


    