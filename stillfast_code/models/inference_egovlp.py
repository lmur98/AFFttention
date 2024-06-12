# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#import sys
#sys.path.append('/home/lmur/stillfast_baseline/stillfast/')
#from stillfast.config.defaults import get_cfg #Import the configurations

import tqdm
import json
import torch
import sys 
sys.path.append('/home/lmur/EgoVLPv2/EgoVLPv2/')
#rom Extract_EgoVLP_dataset import Extract_EgoVLP_Features


import os
import sys
import tqdm
import argparse
import numpy as np
import transformers
import torch
import torch.nn as nn
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser
import pdb
from model.model import FrozenInTime
from transformers import RobertaTokenizer
import model.model as module_arch
    
class EgoVLPv2_inference():
    def __init__(self):
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('-r', '--resume', help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
        args.add_argument('-c', '--config', default='/home/lmur/EgoVLPv2/EgoVLPv2/configs/eval/my_mq.json', type=str, help='config file path (default: None)')
        args.add_argument('-s', '--sliding_window_stride', default=-1, type=int, help='test time temporal augmentation, repeat samples with different start times.')
        args.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='split to evaluate on.')
        args.add_argument('--batch_size', default=1, type=int, help='size of batch')
        args.add_argument('--save_dir', default = '/home/lmur/stillfast_baseline/stillfast/saved_models/STILL_FAST_R50_X3DM_EGO4D_v1.yaml', help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
        args.add_argument('--cuda_base', default = 'cuda:0', help="in form cuda:x")
    
        config = ConfigParser(args, test=True, eval_mode='mq')
        # hack to get sliding into config
        args = args.parse_args()
        config._config['sliding_window_stride'] = args.sliding_window_stride
        self.model = config.initialize('arch', module_arch)
        self.model.eval()
        print(self.model, 'EgoVLP loaded properly :)')

    
    # extract clip features
    #data['video'] = data['video'].to(device)
    #print(data['video'].shape, 'input shape after the dataloader')
            
    #data_batch = {'video': data['video'], 'text': None} #{'video': data['video'][start:end,]}
    #embeds = model.infer(data_batch, task_names = 'EgoNCE', return_embeds = True, ret = {})['video_embeds']
            