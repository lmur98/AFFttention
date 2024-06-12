import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List
from torchvision import transforms
import pickle
import pandas as pd
import cv2
import math
from tqdm import tqdm
import PIL
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import sys
sys.path.append('.../stillfast_baseline/stillfast/')
from stillfast.datasets.sta_hlmdb import Ego4DHLMDB
from stillfast.datasets.ego4d_sta_still import Ego4dShortTermAnticipationStill
from stillfast.datasets.ego4d_sta_still_video import Ego4DHLMDB_STA_Still_Video
from .build import DATASET_REGISTRY

class EK55_HLMDB_STA_Still_Video(Ego4DHLMDB):
    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())

                return Image.open(io.BytesIO(data))
    
    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames: #MOOOOODIFIED!!, THE LMDB STRUCTURE IS DIFFERENT
                    #data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())
                    data = txn.get(frame.encode())
                    out.append(Image.open(io.BytesIO(data)))
            return out
    
    def get_mp3_batch(self, video_id: str, sampled_frames: List[int]) -> List[np.ndarray]:
        out = []
        video_path = os.path.join('.../EPIC-KITCHENS/videos/30fps', video_id + '.MP4')
        #video_path = os.path.join('/home/lmur/EPIC-KITCHENS/videos/original/' + str(video_id[0:3]) + '/videos', video_id + '.MP4')
    
        frames = []
        success_idxs = []
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened())
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frames)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Frame not found', sampled_frames, video_id)
        
        """
        curr_frame = sampled_frames[0]
        while curr_frame <= sampled_frames[-1]:
            ret, frame = cap.read()
            if ret and curr_frame in sampled_frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                success_idxs.append(curr_frame)
            else:
                pass
                # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
            curr_frame += 1
        """
        cap.release()
        # print("Sampled Frames Finally: ", success_idxs)
        #frames_torch = [self.transform(frame) for frame in frames]
        #frames_torch = torch.stack(frames_torch, dim=0)
        return frame
 
@DATASET_REGISTRY.register()
class EpicKitchens_ShortTermAnticipation(torch.utils.data.Dataset):
    """
    Ego4d Short Term Anticipation Still Dataset
    """

    def __init__(self, cfg, split):
        self._split = split
        
        self._still_frames_path = '.../EPIC-KITCHENS/still_frames_STA_ek100_v2'
        self.convert_tensor = transforms.ToTensor()
        self._fast_EK55_hlmdb = EK55_HLMDB_STA_Still_Video('.../EPIC-KITCHENS/ek100/lmdbs/frames', readonly=True, lock=False)
        #self._still_EK55_hlmdb = EK55_HLMDB_STA_Still_Video('/home/lmur/EPIC-KITCHENS/videos/30fps', readonly=True, lock=False)
        #existing_parents = self._fast_EK55_hlmdb.get_existing_keys()
        #print('The existing parents are', existing_parents)
        
        self._load_data()
        
        self.class_names_pkl = '/home/furnari/data/EK55-STA/class_names.pkl'
        self.class_names = pd.read_pickle(self.class_names_pkl)
        self.verb_classes = self.class_names['verbs']
        self.noun_classes = self.class_names['objects']
        
        self.remap_noun_classes()
        self.remap_verb_classes()
        #self.get_class_frequency()
        self.cfg = cfg
        
        self._assign_groups_based_on_resolutions()
        self.predicted_AFF_file = '/home/lmur/catania_lorenzo/EK_data_extracted/AFF_results/predicted_AFF_only_nodes_weighted_MAX_normalized_softmaxT1_TEXT_AND_VIS_4096.npy'
        self.predicted_AFF = np.load(self.predicted_AFF_file, allow_pickle=True).tolist()

    def _get_AFF_labels(self, video_id, frame_number):
        for f in self.predicted_AFF:
            if f['v_id'] == video_id and f['frame'] == frame_number:
                #For weighted distribution
                aff_N_GT = f['AFF_N_top3']
                aff_V_GT = f['AFF_V_top3']
                return {'AFF_N_GT': aff_N_GT, 'AFF_V_GT': aff_V_GT}
        
    def _assign_groups_based_on_resolutions(self):
        clmap = {a['video_id']:f"{1920}_{1080}" for a in self._annotations['annotations']}
        self.groups = [clmap[a['video_id']] for a in self._annotations['annotations']]

    
    def get_class_frequency(self):
        self.nouns_count = np.zeros(210)
        self.verbs_count = np.zeros(210)

        for annot_i in self._annotations['annotations']:
            for n_class in annot_i['objects']['noun_category_id']:
                self.nouns_count[n_class] += 1
            for v_class in annot_i['objects']['verb_category_id']:
                self.verbs_count[v_class] += 1
        for n, noun in enumerate(self.nouns_count):
            print('The', n, 'appears', noun, 'times')
        for v, verb in enumerate(self.verbs_count):
            print('The', v, 'appears', verb, 'times')

    
    def remap_noun_classes(self):
        self.noun_grouped_classes = {'drinks': ['water', 'beer', 'coke', 'juice'],
                                    'foods': ['tortilla', 'cereal', 'yoghurt', 'grater', 'bacon', 'paste:garlic', 'processor:food', 'butter', 'tofu', 'cream',
                                            'honey', 'noddle', 'hummus', 'nutella', 'omelette', 'paella', 'sanwich', 'pancake', 'leftover', 'soup', 'salami',
                                            'biscuit', 'beef', 'food', 'burger:tuna'],
                                    'chicken': ['chicken', 'turkey'],
                                    'vegetables': ['vegetable', 'grape', 'banana',  'lemon', 'avocado', 'blueberry', 'nut:pine', 'oat', 'raisin', 'broccoli', 'berry',
                                                'celery', 'kiwi', 'coconut', 'cabagge', 'spinach', 'fruit', 'dressing:salad', 'pea', 'seed', 'apple', 'pear', 'melon', 'grass:lemon',
                                                'caper', 'pineapple', 'leaf', 'vegetable', 'cabbage', 'bean:green'],
                                    'towel': ['towel', 'towel:kitchen'],
                                    'condiments': ['ginger', 'oregano', 'pesto', 'powder:coconut', 'grinder', 'parsley', 'herb', 'yeast', 'coriander', 'turmeric', 'curry',
                                                'paprika', 'mayonnaise', 'basil', 'cinnamon', 'crisp', 'mustard', 'syrup', 'cumin', 'nesquik', 'spice', 'chilli', 'mint', 'flake:chilli'],
                                    'others': ['cooker:slow', 'mat:sushi', 'cover', 'heat', 'leek', 'light', 'kitchen', 'roll', 'floor', 'scale', 'shelf', 'timer', 'switch', 'cap', 'support',
                                            'content', 'fan:extractor', 'plug', 'button', 'base', 'tissue', 'time', 'clip', 'opener:bottle', 'sock', 'dust', 'desk', 'mesh', 'boxer',
                                            'squeezer:lime', 'phone', 'shell:egg', 'scrap', 'sticker', 'dumpling', 'watch', 'whetstone', 'candle', 'control:remote', 'shaker:pepper', 'alarm',
                                            'recipe', 'wire', 'label', 'presser', 'rest', 'table', 'poster', 'rubber', 'rug', 'sheets', 'tube', 'finger', 'juicer:lime', 'stock', 'apron',
                                            'pin:rolling', 'can', 'mushroom', 'fish', 'lime', 'grill', 'ingredient', 'holder:filter', 'cake', 'almond'],
                                    'appliances': ['machine:washing', 'blender', 'heater', 'toaster', 'cloth', 'tablet'],
                                    'leftover': ['peeler:potato', 'leftover'],
                                    'soap': ['soap', 'powder:washing'],
                                    'utensils': ['utensil', 'masher'],
                                    'foil': ['foil', 'wrap:plastic', 'wrap'],
                                    'coffee': ['tea', 'mocha', 'coffee'],
                                    'pestle': ['pestle', 'mortar'],
                                    'cucumber': ['cucumber', 'squash'],
                                    'onion': ['onion', 'onion:spring']}
        self.new_noun_classes = {0: 'drinks', 1: 'foods', 2: 'chicken', 3: 'vegetables', 4: 'towel', 5: 'condiments', 6: 'others', 7: 'appliances', 8: 'leftover', 
                                 9: 'soap', 10: 'utensils', 11: 'foil', 12: 'coffee', 13: 'pestle', 14: 'cucumber', 15: 'onion'}
        new_index = 16
        for key_n, value_n in self.noun_classes.items():
            found = False
            for key, value in self.noun_grouped_classes.items():
                if value_n in value:
                    found = True
            if not found:
                self.new_noun_classes[new_index] = value_n
                new_index += 1
        
        self.indexer_nouns = {}
        for key_n, value_n in self.noun_classes.items():
            for key, value in self.new_noun_classes.items():
                if value_n == value:
                    self.indexer_nouns[key_n] = key
            idx = 0
            for group, values_group in self.noun_grouped_classes.items():
                if value_n in values_group:
                    self.indexer_nouns[key_n] = idx
                idx += 1
                
        print('The initial length is', len(self._annotations['annotations']))
        #Remapp and filter the annotations, we remove the 'others' class
        self.new_annots = {}
        self.new_annots = {'videos': [], 'annotations': []}
        for annot in self._annotations['annotations']:
            old_noun_category = annot['objects']['noun_category_id']
            new_noun_category = [self.indexer_nouns[n] for n in old_noun_category]
            if 6 in new_noun_category: #We discard 
                continue
            else:
                filter_noun_category = [n if n < 6 else n - 1 for n in new_noun_category]
                annot['objects']['noun_category_id'] = filter_noun_category
                self.new_annots['annotations'].append(annot)
                            
        self.filter_noun_classes = {}
        for key in self.new_noun_classes.keys():
            if key < 6:
                self.filter_noun_classes[key] = self.new_noun_classes[key]
            elif key == 6:
                continue
            else:
                self.filter_noun_classes[key - 1] = self.new_noun_classes[key]
        self.noun_classes = self.filter_noun_classes.copy()
        print(self.noun_classes)
        with open('.../EK_data_extracted/remapped_nouns.json', 'w') as f:
            json.dump(self.noun_classes, f)
        self._annotations = self.new_annots.copy()
        #print(self.noun_classes)
        #print('Filtered by the nouns', len(self._annotations['annotations']))
        
        #for i in range(len(self.new_annots['annotations'])):
        #    print('ANNOTATION')
        #    new = self.new_annots['annotations'][i]['objects']['noun_category_id']
        #    old = self._annotations['annotations'][i]['objects']['noun_category_id']
        #    print('NEW', new, 'OLD', old)
    
    
    def remap_verb_classes(self):
        verbs_to_remove = ['sharpen', 'use', 'pat', 'drink', 'choose', 'wear', 'carry', 'twist', 'sweep', 'rub', 'unwrap', 'attach',
                           'lower', 'unscrew', 'squirt', 'load', 'unroll', 'water', 'do', 'slide', 'unplug', 'fix', 'spill', 'blow',
                           'sit-on', 'flush', 'swirl', 'stick', 'pet-down', 'realize', 'reverse', 'wear', 'set', 'scrub', 'cook', 
                           'finish', 'look', 'serve', 'rip', 'flatten', 'level', 'prepare', 'stack', 'hang', 'knead', 'pull', 'form', 'tip', 'eat', 'stretch']
        self.filtered_verbs = {}
        self.indexer_verbs = {}
        self.removed_indices = []
        new_key = 0
        
        for key, value in self.verb_classes.items():
            if value in verbs_to_remove:
                self.removed_indices.append(key)
                continue
            self.filtered_verbs[new_key] = value
            self.indexer_verbs[key] = new_key
            new_key += 1
        
        self.new_annots = {}
        self.new_annots = {'videos': [], 'annotations': []}
        for annot in self._annotations['annotations']:
            old_verb_category = annot['objects']['verb_category_id']
            if any(v in self.removed_indices for v in old_verb_category):
                continue
            else:
                filter_verb_category = [self.indexer_verbs[v] for v in old_verb_category]
                annot['objects']['verb_category_id'] = filter_verb_category
                self.new_annots['annotations'].append(annot)
        self.verb_classes = self.filtered_verbs.copy()

        with open('.../EK_data_extracted/remapped_verbs.json', 'w') as f:
            json.dump(self.verb_classes, f)
        self._annotations = self.new_annots.copy()

    
                
    def from_panda_to_json(self, panda_data):
        annotations = []
        unique_indices = panda_data.index.unique()
        for u_ind  in unique_indices:
            rows_with_eid = panda_data.loc[[u_ind]]
            annot_i = {'eid': u_ind,
                       'video_id': rows_with_eid['video_id'].values[0],
                       'timestamp': rows_with_eid['timestamp'].values[0],
                       'objects': {'noun_category_id': [],
                                   'verb_category_id': [],
                                   'boxes': [],
                                   'time_to_contact': []}}
            
            for i, row in rows_with_eid.iterrows():
                if (row['x2'] - row['x1']) * (row['y2'] - row['y1']) > 0:    
                    annot_i['objects']['noun_category_id'].append(row['object_class'])
                    annot_i['objects']['verb_category_id'].append(row['verb_class'])
                    annot_i['objects']['boxes'].append(np.array([row['x1'], row['y1'], row['x2'], row['y2']])) 
                    annot_i['objects']['time_to_contact'].append(row['time_to_contact'])
                else:
                    print('Empty bbox')

            annotations.append(annot_i)

        #Save the annotations in a pickle
        general_dir = '.../data_extracted/EK_55_STA'
        pickle_dir = os.path.join(general_dir, self._split + '_annotations.pkl')
        with open(pickle_dir, 'wb') as file:
            pickle.dump(annotations, file)
       
        
    def _load_lists_pickle(self, annot_path):
        """ Load lists. """
    
        res = {
            'videos': {},
            'annotations': []
        }
        res['annotations'] = pd.read_pickle(annot_path)
        return res

    def _load_data(self):
        """
        Load frame paths and annotations from files
        Args:
            cfg (CfgNode): config
        """
        if self._split == "train":
            annot_path = '/home/furnari/data/EK55-STA/training.pkl'
            pickle = pd.read_pickle(annot_path)
            self.pickle_file = pickle
            annot_path = '/home/lmur/catania_lorenzo/data_extracted/EK_55_STA/train_annotations.pkl'
            self._annotations = self._load_lists_pickle(annot_path)
            print('The annotations are', len(self._annotations['annotations']))
        elif self._split == "val":
            #annot_path = '/home/furnari/data/EK55-STA/validation.pkl'
            annot_path = '/home/lmur/catania_lorenzo/data_extracted/EK_55_STA/val_annotations.pkl'
            self._annotations = self._load_lists_pickle(annot_path)
            print('The annotations are', len(self._annotations['annotations']))
        elif self._split == "test":
            annot_path = '/home/lmur/catania_lorenzo/data_extracted/EK_55_STA/val_annotations.pkl'
            self._annotations = self._load_lists_pickle(annot_path)

    def __len__(self):
        """ Get the number of samples. """
        return len(self._annotations['annotations'])
    
    
    def _sample_fast_frames(self, frame):
        """ Sample frames from a video. """
        stride = 1 #4 we see 2 secs, 2 we see 1 sec, 1 we see 0,5 sec
        frames = (frame - np.arange(16 * stride, step=stride,)[::-1])
        frames[frames < 1] = 1 #Offset
        frames = frames.astype(int)
        return frames
    
    def _load_still_fast_frames(self, video_id, timestamp):
        """ Load frames from video_id and frame_number """
        frame_number = int(timestamp * 30 + 1) #VIDEOS ARE AT 30 FPS

        still_img = Image.open(os.path.join(self._still_frames_path, f"{video_id}_{frame_number:010d}.jpg"))

        fast_frames_list = self._sample_fast_frames(frame_number)
        new_frames_list = []
        for f in fast_frames_list:
            new_frames_list.append(f"{video_id}_frame_{f:010d}.jpg")
        fast_imgs = self._fast_EK55_hlmdb.get_batch('rgb', new_frames_list)
        
        return still_img, frame_number, fast_imgs, fast_frames_list
    
    def _load_annotations_EK(self, idx):
        """ Load annotations for the idx-th sample. """
        # get the idx-th annotation
        ann = self._annotations['annotations'][idx]
        uid = ann['eid']

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann["video_id"]
        
        if 'objects' in ann:
            gt_noun_labels = np.array(ann['objects']['noun_category_id'])
            gt_verb_labels = np.array(ann['objects']['verb_category_id'])
            gt_ttc_targets = np.array(ann['objects']['time_to_contact'])
            gt_boxes = np.asarray(ann['objects']['boxes']) * (1920, 1080, 1920, 1080) #x1y1x2y2 
        else:
            gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None
        return uid, video_id, ann['timestamp'], gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets
   
    def __getitem__(self, idx):
        """ Get the idx-th sample. """
        uid, video_id, timestamp, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations_EK(idx)
        still_img, still_frame, fast_imgs, frames_list = self._load_still_fast_frames(video_id, timestamp)
        still_img_torch = self.convert_tensor(still_img)
        fast_imgs_torch = torch.stack([self.convert_tensor(img) for img in fast_imgs], dim=1)
        ch, T, h_img, w_img = fast_imgs_torch.shape
        
        # FIXME: this is a hack to make the dataset compatible with the original Ego4d dataset
        # This could create problems when producing results on the test set and sending them to the
        # evaluation server.
        verb_offset = 1 #ROI V2
        
        targets = {
            'boxes': torch.from_numpy(gt_boxes),
            'noun_labels': torch.Tensor(gt_noun_labels).long() + 1,
            'verb_labels': torch.Tensor(gt_verb_labels).long() + verb_offset,
            'ttc_targets': torch.Tensor(gt_ttc_targets),
        } if gt_boxes is not None else None
        
        
        frame_number = round(timestamp * 30) + 2
        aff_prior = self._get_AFF_labels(video_id, frame_number)
        if aff_prior is None:
            print("WARNING: AFF GT NOT FOUND FOR VIDEO: ", video_id, " FRAME: ", frame_number)
            aff_N = np.ones((self.cfg.MODEL.NOUN_CLASSES + 1)) #Uniform
            aff_V = np.ones((self.cfg.MODEL.VERB_CLASSES + verb_offset))
            #aff_N[gt_noun_labels[0] + 1] = 1.0
            #aff_V[gt_verb_labels[0] + verb_offset] = 1.0
            targets['AFF_N_GT'] = torch.Tensor(aff_N)
            targets['AFF_V_GT'] = torch.Tensor(aff_V)
        else:
            targets['AFF_N_GT'] = torch.Tensor(aff_prior['AFF_N_GT'])
            targets['AFF_V_GT'] = torch.Tensor(aff_prior['AFF_V_GT'])

        
        return {'still_img': still_img_torch, 'fast_imgs': fast_imgs_torch, 'targets': targets, 
                'video_id': video_id, 'timestamp': timestamp, 'still_frame': still_frame, 'uids': uid}

if __name__ == '__main__':
    dataset = EpicKitchens_ShortTermAnticipation(cfg = None, split = 'train')
    print(len(dataset), '*************************************')
    print(len(dataset.verb_classes), 'VERBS')
    print(len(dataset.noun_classes), 'NOUNS')
    samples_to_show = [59, 5798, 4312, 162, 2598, 867, 256, 13450, 13451, 13452]
    frames_dir = '/home/lmur/EPIC-KITCHENS/still_frames_STA_ek100_v2'
    for id in samples_to_show:
        sample = dataset[id]
    #for i, sample in tqdm(enumerate(dataset)):    
        bboxes = sample['targets']['boxes']
        noun_label = sample['targets']['noun_labels'][0].numpy()
        verb_label = sample['targets']['verb_labels'][0].numpy()
        still_img = sample['still_img']
        fast_imgs = sample['fast_imgs']
        print(still_img.shape, fast_imgs.shape, 'shapes')
        print(bboxes)
        #Save the still image
        #img_name = f'{sample["video_id"]}_{sample["still_frame"]:010d}.jpg'
        #img_path = os.path.join(frames_dir, img_name)
        #still_img = cv2.cvtColor(still_img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(img_path, still_img)
        

 
        new_width = 456
        new_height = 256
        
        transform_still = transforms.Compose([transforms.ToPILImage()])
        transform_tensor = transforms.Compose([transforms.Resize((new_height, new_width)), transforms.ToTensor()])
        still_img_show = transform_still(still_img)
        draw = ImageDraw.Draw(still_img_show)
        x1, y1, x2, y2 = bboxes[0] 
        draw.rectangle([(x1, y1), (x2, y2)], outline='green', width = 8)
        
        fast_img_1 = fast_imgs[:, 0, :, :]
        fast_img_4 = fast_imgs[:, 3, :, :]
        fast_img_8 = fast_imgs[:, 7, :, :]
        fast_img_12 = fast_imgs[:, 11, :, :]
        fast_img_16 = fast_imgs[:, -1, :, :]
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        
        axs[0, 0].imshow(transform_tensor(still_img_show).permute(1, 2, 0))
        axs[0, 0].set_title('Still Image')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(fast_img_1.permute(1, 2, 0))
        axs[0, 1].set_title('Fast Image 1')
        axs[0, 1].axis('off')
        
        axs[0, 2].imshow(fast_img_4.permute(1, 2, 0))
        axs[0, 2].set_title('Fast Image 4')
        axs[0, 2].axis('off')
        
        axs[1, 0].imshow(fast_img_8.permute(1, 2, 0))
        axs[1, 0].set_title('Fast Image 8')
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(fast_img_12.permute(1, 2, 0))
        axs[1, 1].set_title('Fast Image 12')
        axs[1, 1].axis('off')
        
        axs[1, 2].imshow(fast_img_16.permute(1, 2, 0))
        axs[1, 2].set_title('Fast Image 16')
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        #Save the images
        plt.savefig(os.path.join('/home/lmur/catania_lorenzo/data_extracted/EK_55_STA/check_', 'aa' + str(sample['timestamp']) + '_still.jpg'), dpi = 300)
        plt.close()
   