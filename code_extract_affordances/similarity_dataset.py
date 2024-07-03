import numpy as np
import collections
import torch
import os
import glob
from PIL import Image, ImageOps
import h5py
import tqdm
import itertools
import torch
import json
import torchvision.transforms as transforms


#-------------------------------------------------------------------------------------------#

# Base class that implements all selection strategies
class Similarity_Frames_Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.still_frames_path = '/home/furnari/data/ego4d/v2-15-02-23/object_frames/'
        self.split = split
        if self.split == 'train':
            self.labels_train = '/home/lmur/hum_obj_int/stillfast/output_topo_graphs/similarity_frames_per_video'
            self.labels_val = '/home/lmur/hum_obj_int/stillfast/output_topo_graphs_VAL/similarity_frames_per_video'
        elif self.split == 'val':
            self.labels_val = '/home/lmur/hum_obj_int/stillfast/output_topo_graphs_VAL/similarity_frames_per_video'
        self.global_labels = self.join_all_labels()
        seed = 8275 if self.split=='val' else None
        self.rs = np.random.RandomState(seed)
        self.transform = self.transform_function()

    
    def join_all_labels(self):
        labels = []
        if self.split == 'train':
            for file in glob.glob(self.labels_train + '/*.json'):
                data = json.load(open(file, 'r'))
                labels.extend(data)
            for file in glob.glob(self.labels_val + '/*.json'):
                data = json.load(open(file, 'r'))
                labels.extend(data)
        elif self.split == 'val':
            for file in glob.glob(self.labels_val + '/*.json'):
                data = json.load(open(file, 'r'))
                labels.extend(data)
        return labels

    def load_img(self, video_id, frame_n):
        img_path = os.path.join(self.still_frames_path, f"{video_id}_{frame_n:07d}.jpg")
        return Image.open(img_path).convert('RGB')

    def transform_function(self):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if self.split == 'train':
            transform = transforms.Compose([transforms.Resize(256), #Downscales the image in a x4 factor aprox
                                            transforms.CenterCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        elif self.split == 'val':
            transform = transforms.Compose([transforms.Resize(256), #Downscales the image in a x4 factor aprox
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
        return transform

    # subclasses should decide how to actually pick among selection strategies
    # for positive and negative samples
    def select_positive(self, video_id, frameA, similar_frames):
        if len(similar_frames) == 0: #If there are no similar frames, we flip the frame
            frame_n = frameA
            imgB = self.transform(self.load_img(video_id, frame_n))
            imgB = transforms.functional.hflip(imgB)
        else:
            frame_n = similar_frames[self.rs.randint(0, len(similar_frames))] #Select randomly one of the similar frames
            imgB = self.transform(self.load_img(video_id, frame_n))
        return imgB
            
    def select_negative(self, video_id, dissimilar_frames):
        if len(dissimilar_frames) == 0: #If there are no dissimilar frames, we select a random frame from the dataset
            new_sample = self.global_labels[self.rs.randint(0, len(self.global_labels))]
            new_video_id = new_sample['video_id']
            new_frame_n = new_sample['frame']
            imgB = self.transform(self.load_img(new_video_id, new_frame_n))
        else:
            frame_n = dissimilar_frames[self.rs.randint(0, len(dissimilar_frames))] #Select randomly one of the dissimilar frames
            imgB = self.transform(self.load_img(video_id, frame_n))
        return imgB


    def __getitem__(self, index):
        #Read the extracted labels
        video_id = self.global_labels[index]['video_id']
        frameA = self.global_labels[index]['frame']
        similar_frames = self.global_labels[index]['similar']
        dissimilar_frames = self.global_labels[index]['dissimilar']

        #Load the images according to its label
        imgA = self.transform(self.load_img(video_id, frameA))
        label = 1 if self.rs.rand()<0.5 else 0
        if label == 1:
            imgB = self.select_positive(video_id, frameA, similar_frames)
        else:
            imgB = self.select_negative(video_id, dissimilar_frames)
        
        # arbitrary order for frameA, frameB
        if self.rs.rand()<0.5:
            imgA, imgB = imgB, imgA

        return {'imgA':imgA, 'imgB':imgB, 'label':label}

    def __len__(self):
        return len(self.global_labels)


#------------------------------------------To visualize the dataset--------------------------------------#


def unnormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    u_tensor = tensor.clone()
    
    def _unnorm(t):
        for c in range(3):
            t[c].mul_(std[c]).add_(mean[c])

    if u_tensor.dim()==4:
        [_unnorm(t) for t in u_tensor]
    else:
        _unnorm(u_tensor)
    
    return u_tensor

from PIL import Image, ImageOps
def add_border(img, color, sz=128):
    img = transforms.ToPILImage()(img)
    img = ImageOps.expand(img, border=5, fill=color)
    img = img.resize((sz, sz))
    img = transforms.ToTensor()(img)
    return img

import cv2
import sys
def show_wait(img, idx, T=0, win='image', sz=None, save=None):
    shape = img.shape
    img = transforms.ToPILImage()(img)
    if sz is not None:
        H_new = int(sz/shape[2]*shape[1])
        img = img.resize((sz, H_new))

    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    cv2.imwrite(os.path.join('/home/lmur/hum_obj_int/stillfast/extract_affordances/labels_show', str(idx) + '.jpg'), open_cv_image)
    print('writeeen')
    

if __name__=='__main__':
    from torchvision.utils import make_grid

    dataset = Similarity_Frames_Dataset('train')
    #Make the order random
    dataset.global_labels = dataset.rs.permutation(dataset.global_labels)
    print('the len of the training is', len(dataset))

    viz = []
    for idx, entry in enumerate(dataset):

        instance = dataset[idx]
        imgA = unnormalize(instance['imgA'])
        imgB = unnormalize(instance['imgB'])

        color = 'green' if instance['label']==1 else 'red'
        imgA, imgB = add_border(imgA, color), add_border(imgB, color)
        viz += [imgA, imgB]

        print('Image')

        if len(viz)==20:
            viz = viz[0::2] + viz[1::2]
            grid = make_grid(viz, nrow=10)
            show_wait(grid, idx)
            viz = []
            print ('-'*10)