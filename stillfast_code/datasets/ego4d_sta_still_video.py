import torch
import json
import os.path
import numpy as np
from PIL import Image
import io
from typing import List

from .build import DATASET_REGISTRY
import sys
sys.path.append('../stillfast_baseline/stillfast/')
from stillfast.config.defaults import get_cfg
from stillfast.datasets.sta_hlmdb import Ego4DHLMDB
from stillfast.datasets.ego4d_sta_still import Ego4dShortTermAnticipationStill
from stillfast.datasets.ego4d_sta_hands_still_video import Ego4d_withHands_ShortTermAnticipationStillVideo
from stillfast.datasets import StillFastImageTensor
from torchvision import transforms
import tqdm 


class Ego4DHLMDB_STA_Still_Video(Ego4DHLMDB):
    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())

                return Image.open(io.BytesIO(data))
    
    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id,frame_number=frame).encode())
                    out.append(Image.open(io.BytesIO(data)))
            return out
        
    def get_mp3_batch(self, video_id: str, sampled_frames: List[int]) -> List[np.ndarray]:
        out = []
        video_path = os.path.join('.../EPIC-KITCHENS/videos/30fps', video_id + '.mp4')
        frames = []
        success_idxs = []
        cap = cv2.VideoCapture(video_path)
        assert (cap.isOpened())
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frames[0] - 1)

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
        cap.release()
        # print("Sampled Frames Finally: ", success_idxs)
        #frames_torch = [self.transform(frame) for frame in frames]
        #frames_torch = torch.stack(frames_torch, dim=0)
        return frames

# TODO: refactor as reconfigurable
@DATASET_REGISTRY.register()
class Ego4dShortTermAnticipationStillVideo(Ego4dShortTermAnticipationStill):
    """
    Ego4d Short Term Anticipation StillVideo Dataset
    """

    def __init__(self, cfg, split):
        super(Ego4dShortTermAnticipationStillVideo, self).__init__(cfg, split)
        #Only for extract EgoVLP features
        norm_mean=(0.485, 0.456, 0.406)
        norm_std=(0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            normalize,
        ])
        self._fast_hlmdb = Ego4DHLMDB_STA_Still_Video(self.cfg.EGO4D_STA.FAST_LMDB_PATH, readonly=True, lock=False)
        self.predicted_AFF_file = '.../results/predicted_AFF_only_nodes_weighted_MAX_normalize_softnormT2.npy'
        self.predicted_AFF = np.load(self.predicted_AFF_file, allow_pickle=True).tolist()
        #print('The number of predicted AFF is:', len(self.predicted_AFF))
        
    def _load_frames_lmdb(self, video_id, frames):
        """ Load images from lmdb. """
        imgs = self._fast_hlmdb.get_batch(video_id, frames)
        return imgs

    def _sample_frames(self, frame):
        """ Sample frames from a video. """
        frames = (frame - np.arange(self.cfg.DATA.FAST.NUM_FRAMES * self.cfg.DATA.FAST.SAMPLING_RATE,step=self.cfg.DATA.FAST.SAMPLING_RATE,)[::-1])
        frames[frames < 0] = 0
        frames = frames.astype(int)
        return frames

    def _load_still_fast_frames(self, video_id, frame_number):
        """ Load frames from video_id and frame_number """
        frames_list = self._sample_frames(frame_number)
        fast_imgs = self._load_frames_lmdb(
                video_id, frames_list
            )

        still_img = self._load_still_frame(video_id, frame_number)
        return still_img, fast_imgs, frames_list
    
    def read_narrations(self, video_id, frame):
        narrations_json = json.load(open(os.path.join('.../hum_obj_int/narrations_dict', f"{video_id}.json")))
        
        if 'narration_pass_1' in narrations_json:
            clip_narrations_1 = narrations_json['narration_pass_1']['narrations']
            clip_narrations_1 = sorted(clip_narrations_1, key=lambda k: k['timestamp_frame'])
            narrator_1 = True
        else:
            narrator_1 = False
        if 'narration_pass_2' in narrations_json:
            clip_narrations_2 = narrations_json['narration_pass_2']['narrations']
            clip_narrations_2 = sorted(clip_narrations_2, key=lambda k: k['timestamp_frame'])
            narrator_2 = True
        else:
            narrator_2 = False
        
        #We need to find the narration that corresponds to each frame
        narrations = {'frame_number': frame, 'text': None}
        if not(narrator_1) and not(narrator_2):
            return narrations
        if narrator_1:
            for n in range(len(clip_narrations_1) - 1):
                narration = clip_narrations_1[n]['narration_text']
                start_frame = clip_narrations_1[n]['timestamp_frame']
                end_frame = clip_narrations_1[n + 1]['timestamp_frame']
                if frame >= start_frame and frame < end_frame:
                    narrations['narrator_1'] = narration
                    break

        if narrator_2:
            for n in range(len(clip_narrations_2) - 1):
                narration = clip_narrations_2[n]['narration_text']
                start_frame = clip_narrations_2[n]['timestamp_frame']
                end_frame = clip_narrations_2[n + 1]['timestamp_frame']
                if frame >= start_frame and frame < end_frame:
                    narrations['narrator_2'] = narration
                    break

        if 'narrator_1' not in narrations.keys() and 'narrator_2' not in narrations.keys():
            return narrations
        elif 'narrator_1' not in narrations.keys():
            sentence = self.clean_narration(narrations['narrator_2'])
        elif 'narrator_2' not in narrations.keys():
            sentence = self.clean_narration(narrations['narrator_1'])
        else:
            sentence = self.clean_narration(narrations['narrator_1']) + ' and ' + self.clean_narration(narrations['narrator_2'])
        narrations['text'] = sentence
        return narrations
    
    def clean_narration(self, sentence):
        sentence = sentence.replace("#C ", "")
        sentence = sentence.replace("#Unsure", "")
        sentence = sentence.replace("#", "")
        sentence = sentence.replace("C ", "")
        return sentence

    def _get_AFF_labels(self, video_id, frame_number):
        find = False
        for f in self.predicted_AFF:
            if f['v_id'] == video_id and f['frame'] == frame_number:
                #For weighted distribution
                aff_N_GT = f['AFF_N_top3']
                aff_V_GT = f['AFF_V_top3']
                find = True
                break
        if not find:
            return None
        return {'AFF_N_GT': aff_N_GT, 'AFF_V_GT': aff_V_GT}
            
    
    def __getitem__(self, idx):
        """ Get the idx-th sample. """
        uid, video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations(idx)
        
        still_img, fast_imgs, frames_list = self._load_still_fast_frames(video_id, frame_number)

        still_img = self.convert_tensor(still_img)
        fast_imgs_torch = torch.stack([self.convert_tensor(img) for img in fast_imgs], dim=1)
        
        #ch, h, w = still_img.shape
        #scaled_contact_points = pred_contact_points * np.array([w, h])
        #hmap = self.compute_heatmap(scaled_contact_points, (w, h)) #(H, W) !! There is a transpose.
        
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
        } if gt_boxes is not None else {}
        

        #MOD: INCLUDE THE AFF ANNOTATIONS, IN CASE THAT WE USE THEM AS ORACLE
        aff_prior = self._get_AFF_labels(video_id, frame_number)
        
        if aff_prior is None:
            print("WARNING: AFF GT NOT FOUND FOR VIDEO: ", video_id, " FRAME: ", frame_number)
            aff_N = np.ones((self.cfg.MODEL.NOUN_CLASSES + 1)) #Uniform
            aff_V = np.ones((self.cfg.MODEL.VERB_CLASSES + verb_offset))
            targets['AFF_N_GT'] = torch.Tensor(aff_N)
            targets['AFF_V_GT'] = torch.Tensor(aff_V)
        else:
            targets['AFF_N_GT'] = torch.tensor(aff_prior['AFF_N_GT'])
            targets['AFF_V_GT'] = torch.tensor(aff_prior['AFF_V_GT'])

        
        return {'still_img': still_img, 'fast_imgs': fast_imgs_torch, 'targets': targets, 'uids': uid, 'video_id': video_id, 'frame_number': frame_number, 'frames_list': frames_list}
