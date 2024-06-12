import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List
from torchvision import transforms
import pickle

from .build import DATASET_REGISTRY
from stillfast.datasets.utils import get_annotations_weights
import cv2

# TODO: refactor as reconfigurable
@DATASET_REGISTRY.register()
class Ego4dShortTermAnticipationStill(torch.utils.data.Dataset):
    """
    Ego4d Short Term Anticipation Still Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        
        self._still_frames_path = self.cfg.EGO4D_STA.STILL_FRAMES_PATH
        self.convert_tensor = transforms.ToTensor()

        self._load_data(cfg)
        if split == "train":
            self._cleanup()

        self._assign_groups_based_on_resolutions()

        if split == 'train':
            self.weights  = get_annotations_weights(self._annotations)
        with open('.../data_extracted/join_hands_baseline/results/ony_contact_point.pkl', 'rb') as file:
            self.hotspots = pickle.load(file)

    def _cleanup(self):
        removed_boxes = 0
        removed_anns = 0
        anns = self._annotations['annotations']
        self._annotations['annotations'] = []
        for i in range(len(anns)):
            ann = anns[i]
            if 'objects' in ann:
                _obj = []
                for obj in ann['objects']:
                    box = obj['box']
                    if (box[2]-box[0])*(box[3]-box[1]) > 0:
                        _obj.append(obj)
                    else:
                        removed_boxes+=1

                if(len(_obj) > 0):
                    ann['objects'] = _obj
                    self._annotations['annotations'].append(ann)
                else:
                    removed_anns+=1
                    
        print(f"removed {removed_boxes} degenerate objects and {removed_anns} annotations with no objects")


    def _load_lists(self, _list):
        """ Load lists. """
        def extend_dict(input_dict, output_dict):
            for k,v in input_dict.items():
                output_dict[k]=v
            return output_dict

        res = {
            'videos': {},
            'annotations': []
        }
        for l in _list:
            with open(os.path.join(self.cfg.EGO4D_STA.ANNOTATION_DIR,l)) as f:
                j = json.load(f)
            res['videos'] = extend_dict(j['info']['video_metadata'], res['videos'])
            res['annotations'] += j['annotations']
        
        return res

    def _load_lists_pickle(self, annot_path):
        """ Load lists. """
        def extend_dict(input_dict, output_dict):
            for k,v in input_dict.items():
                output_dict[k]=v
            return output_dict

        res = {
            'videos': {},
            'annotations': []
        }
        with open(annot_path, 'rb') as file:
                data = pickle.load(file)
        res['videos'] = extend_dict(data['info']['video_metadata'], res['videos'])
        res['annotations'] += data['annotations']
        
        return res

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files
        Args:
            cfg (CfgNode): config
        """
        if self._split == "train":
            self._annotations = self._load_lists(cfg.EGO4D_STA.TRAIN_LISTS)
        elif self._split == "val":
            self._annotations = self._load_lists(cfg.EGO4D_STA.VAL_LISTS)
        else:
            self._annotations = self._load_lists(cfg.EGO4D_STA.TEST_LISTS)


    def _assign_groups_based_on_resolutions(self):
        clmap = {k:f"{v['frame_width']}_{v['frame_height']}" for k,v in self._annotations['videos'].items()}
        self.groups = [clmap[a['video_id']] for a in self._annotations['annotations']] #V1
        #self.groups = [clmap[a['video_uid']] for a in self._annotations['annotations']] #V2

    def __len__(self):
        """ Get the number of samples. """
        return len(self._annotations['annotations'])

    def _load_still_frame(self, video_id, frame):
        """ Load images from lmdb. """
        still_img = Image.open(os.path.join(self._still_frames_path, f"{video_id}_{frame:07d}.jpg"))

        return still_img
    
    def _load_annotations(self, idx):
        """ Load annotations for the idx-th sample. """
        # get the idx-th annotation
        ann = self._annotations['annotations'][idx]
        uid = ann['uid']

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann["video_id"] #V1
        #video_id = ann["video_uid"] #V2
        frame_number = ann['frame']

        if 'objects' in ann:
            gt_boxes = np.vstack([x['box'] for x in ann['objects']])
            gt_noun_labels = np.array([x['noun_category_id'] for x in ann['objects']])
            gt_verb_labels = np.array([x['verb_category_id'] for x in ann['objects']])
            gt_ttc_targets = np.array([x['time_to_contact'] for x in ann['objects']])
        else:
            #gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None
            gt_boxes = np.array([[0,0,100,100]])
            gt_noun_labels = np.array([0])
            gt_verb_labels = np.array([0])
            gt_ttc_targets = np.array([0])
        return uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets

    def compute_heatmap(self, points, image_size, k_ratio=1.0):
        points = np.asarray(points)
        heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32) #(H, W)
        n_points = points.shape[0]
        for i in range(n_points):
            x = points[i, 0]
            y = points[i, 1]
            col = int(x)
            row = int(y)
            try:
                heatmap[col, row] += 1.0
            except:
                col = min(max(col, 0), image_size[0] - 1)
                row = min(max(row, 0), image_size[1] - 1)
                heatmap[col, row] += 1.0
        k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
        if k_size % 2 == 0:
            k_size += 1
        heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap = heatmap.transpose()

        return heatmap

    def _load_annotations_with_contact_point(self, idx):
        """ Load annotations for the idx-th sample. """
        # get the idx-th annotation
        ann = self._annotations['annotations'][idx]
        uid = ann['uid']

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann["video_id"] #V1
        #video_id = ann["video_uid"] #V2
        frame_number = ann['frame']
        contact_points = None
        for int_hotspot in self.hotspots:
            if f"{video_id}_{frame_number:07d}" == int_hotspot['uid']:
                contact_points = int_hotspot['pred']

        if 'objects' in ann:
            gt_boxes = np.vstack([x['box'] for x in ann['objects']])
            gt_noun_labels = np.array([x['noun_category_id'] for x in ann['objects']])
            gt_verb_labels = np.array([x['verb_category_id'] for x in ann['objects']])
            gt_ttc_targets = np.array([x['time_to_contact'] for x in ann['objects']])
        else:
            gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None

        return uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets, contact_points
    
    def _load_annotations_with_hand(self, idx):
        """ Load annotations for the idx-th sample. """
        # get the idx-th annotation
        ann = self._annotations['annotations'][idx]
        uid = ann['uid']

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann["video_uid"]
        frame_number = ann['frame']
        
        if 'objects' in ann:
            gt_boxes = np.vstack([x['box'] for x in ann['objects']])
            gt_noun_labels = np.array([x['noun_category_id'] for x in ann['objects']])
            gt_verb_labels = np.array([x['verb_category_id'] for x in ann['objects']])
            gt_ttc_targets = np.array([x['time_to_contact'] for x in ann['objects']])
        else:
            gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None

        left_hand = ann['left_hand']
        right_hand = ann['right_hand']
        left_hand.sort(key=lambda x: x['fast_frame']) #Order according the lists of dicts according to the frame
        right_hand.sort(key=lambda x: x['fast_frame'])

        gt_left_hand = [x['box'] for x in left_hand]
        gt_right_hand = [x['box'] for x in right_hand]
        return uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets, gt_left_hand, gt_right_hand
    
    def __getitem__(self, idx):
        """ Get the idx-th sample. """
        #uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets, pred_contact_points = self._load_annotations_with_contact_point(idx)
        uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations(idx)

        img = self._load_still_frame(video_id, frame_number)

        img = self.convert_tensor(img)
        
        
        # FIXME: this is a hack to make the dataset compatible with the original Ego4d dataset
        # This could create problems when producing results on the test set and sending them to the
        # evaluation server.
        if 'v1' not in self.cfg.MODEL.STILLFAST.ROI_HEADS.VERSION:
            verb_offset = 1
        else:
            verb_offset = 0

        targets = {
            'boxes': torch.from_numpy(gt_boxes),
            'noun_labels': torch.Tensor(gt_noun_labels).long()+1,
            'verb_labels': torch.Tensor(gt_verb_labels).long()+verb_offset,
            'ttc_targets': torch.Tensor(gt_ttc_targets),
            #'int_hotspot': torch.Tensor(hmap)
        } if gt_boxes is not None else None

        return {'still_img': img, 'targets': targets, 'uids': uid}
