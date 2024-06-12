from stillfast.models.faster_rcnn import FasterRCNN
from stillfast.models.faster_rcnn_sta import RoiHeadsSTA, RoiHeadsSTAv2
from stillfast.models.faster_rcnn_AFF_sta import RoiHeadsAFFSTAv2
from stillfast.models.faster_rcnn_HANDS_sta import RoiHeads_HANDS_STAv2
from stillfast.models.faster_rcnn_prior_HANDS_sta import RoiHeads_PRIOR_HANDS_STAv2
from .build import MODEL_REGISTRY
from detectron2.config import configurable
from stillfast.models.backbone_utils_stillfast import StillFastBackbone, StillBackbone
from stillfast.transforms import StillFastTransform
from torchvision.models.detection._utils import overwrite_eps
from torchvision._internally_replaced_utils import load_state_dict_from_url
from stillfast.datasets import StillFastImageTensor
from stillfast.models.dino_backbones import DINO_2D_feature_extractor, DINO_2D_3D_feature_extractor, DINO_2D_with_FAST, DINO_2D_with_EgoVLP
from stillfast.models.detr_criterion import SetCriterion
from stillfast.models.detr_backbone import DINO_TimeSformer_for_DETR
from stillfast.models.detr_conv_backbone import R50_for_DETR_backbone, Joiner
from stillfast.models.detr_pos_encoding import PositionEmbeddingLearned, PositionEmbeddingSine
from stillfast.models.detr_transformer import Transformer
from stillfast.models.utils_DETR.misc import NestedTensor
import torch.nn as nn 
import torch
from stillfast.models.detr_matcher import HungarianMatcher
from stillfast.utils.detr_box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
import torch.nn.functional as F
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from torchvision.transforms import ToPILImage

@MODEL_REGISTRY.register()
class StillFast(FasterRCNN):
    @configurable
    def __init__(self, *args, pretrained, transform, **kwargs) -> None:
        super().__init__(*args, pretrained=False, **kwargs)

        #if pretrained:
        #    self.load_faster_rcnn_pretrained_weights(kwargs)

        self.transform = transform

    def load_faster_rcnn_pretrained_weights(self, kwargs):
        print(f"Loading checkpoint from {kwargs['checkpoint_url']}")
        # Load state dict
        state_dict = load_state_dict_from_url(kwargs['checkpoint_url'], progress=True)

        # load pretrained weights for backbone
        missing_keys, unmatched_keys = self.backbone.load_faster_rcnn_pretrained_weights(state_dict)

        # Discard keys that have already been matched
        state_dict = {k: v for k, v in state_dict.items() if k in unmatched_keys}

        # load head only if not replaced
        if not kwargs['replace_head']:
            head_state_dict = {k.replace('roi_heads.',''): v for k, v in state_dict.items() if 'roi_heads' in k}
            m,u = self.roi_heads.load_state_dict(head_state_dict)
            missing_keys += m
            unmatched_keys += u
        else:
            # ignore head weights
            unmatched_keys = [k for k in unmatched_keys if 'roi_heads' not in k]
            print("Skipping roi_heads weights as the head has been replaced")

        # load rpn weights
        rpn_state_dict = {k.replace('rpn.',''):v for k,v in state_dict.items() if 'rpn' in k}
        m, unmatched_keys = self.rpn.load_state_dict(rpn_state_dict, strict=False)

        missing_keys += m
        
        print(f"Missing keys: {missing_keys}")
        print(f"Unmatched keys: {unmatched_keys}")
        
        overwrite_eps(self, 0.0)

    @classmethod
    def from_config(cls, cfg):
        options = super().from_config(cfg)
        if (cfg.MODEL.BRANCH == 'Still'):
            backbone = StillBackbone(cfg.MODEL) #We are using this backbone
        elif (cfg.MODEL.BRANCH == 'StillFast'): 
            backbone = StillFastBackbone(cfg.MODEL)
        elif (cfg.MODEL.BRANCH == 'Dino2D'):
            backbone = DINO_2D_feature_extractor(embedding_size = 384, patch_size = 14, last_layers = [11])
        elif (cfg.MODEL.BRANCH == 'Dino2D_and_3D'):
            backbone = DINO_2D_3D_feature_extractor(cfg)
            backbone.out_channels = 384
        elif (cfg.MODEL.BRANCH == 'Dino2D_with_FAST'):
            backbone = DINO_2D_with_FAST(cfg)
            backbone.out_channels = 384
        elif (cfg.MODEL.BRANCH == 'Dino2D_with_EgoVLP'):
            backbone = DINO_2D_with_EgoVLP(cfg)
            backbone.out_channels = 768
        elif (cfg.MODEL.BRANCH == 'Dino_TimeSfor_768_DETR_head'):
            backbone = DINO_TimeSformer_for_DETR()
            backbone.out_channels = 768
        else:
            raise ValueError(f"Unknown branch: {cfg.MODEL.BRANCH}")


        transform = StillFastTransform(cfg)

        hver = cfg.MODEL.STILLFAST.ROI_HEADS.VERSION

        if  hver == 'v1':
            roi_heads = RoiHeadsSTA(cfg)
        elif hver == 'v2':
            roi_heads = RoiHeadsSTAv2(cfg)
        elif hver == 'aff':
            roi_heads = RoiHeadsAFFSTAv2(cfg)
        elif hver == 'hands_v2':
            roi_heads = RoiHeads_HANDS_STAv2(cfg)
        elif hver == 'prior_hands_v2':
            roi_heads = RoiHeads_PRIOR_HANDS_STAv2(cfg)
        elif hver == 'DETR':
            roi_heads = DETR_head(cfg)
        else:
            raise ValueError(f"Unknown version of RoiHeads: {hver}")
    
        options.update({
            'roi_heads': roi_heads,
            'backbone': backbone,
            'transform': transform
        })
            
        return options 

    def forward(self, batch):
        images = [StillFastImageTensor(a, b) for a,b in zip(batch['still_img'], batch['fast_imgs'])]
        targets = batch['targets'] if 'targets' in batch else None
        return super().forward(images, targets)
