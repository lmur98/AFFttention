import torch
import torch.nn as nn
import torch.nn.functional as F

class Dino_2D_extractor(nn.Module):
    def __init__(self):
        super(Dino_2D_extractor, self).__init__()
        self.dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for name, parameter in self.dino_extractor.named_parameters():
            parameter.requires_grad_(False)
        
    def forward(self, img):
        # --- Extract deep descriptors from DINO-vit --- #
        w_i, h_i = img.shape[-2:]
        with torch.no_grad():
            dino_output = self.dino_extractor.forward_features(img)
            x_patch = dino_output["x_norm_patchtokens"].detach()
            x_clss = dino_output["x_norm_clstoken"].detach().unsqueeze(1)
        return x_clss, x_patch, w_i, h_i


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP_Dino_single(nn.Module):
    def __init__(self, aff_verbs, aff_nouns):
        super(MLP_Dino_single, self).__init__()
        self.aff_verbs = aff_verbs
        self.aff_nouns = aff_nouns
        
        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.aff_fc_nouns = nn.Sequential(nn.Linear(self.vit_feat_dim, self.vit_feat_dim), nn.ReLU(True), nn.Dropout(0.5),
                                          nn.Linear(self.vit_feat_dim, self.aff_nouns))
        self.aff_fc_verbs = nn.Sequential(nn.Linear(self.vit_feat_dim, self.vit_feat_dim), nn.ReLU(True), nn.Dropout(0.5),
                                          nn.Linear(self.vit_feat_dim, self.aff_verbs))
    
    def forward(self, dino_cls_token):
        # --- Extract deep descriptors from DINO-vit --- #
        x = dino_cls_token
        nouns_logits = self.aff_fc_nouns(x)
        verbs_logits = self.aff_fc_verbs(x)
        
        return {'1D_verb_prior': verbs_logits, '1D_noun_prior': nouns_logits,
                'aff_2D_map_VERBS': None, 'aff_2D_map_NOUNS': None}


class Locate_Aff(nn.Module):
    def __init__(self, aff_verbs, aff_nouns):
        super(Locate_Aff, self).__init__()
        self.aff_verbs = aff_verbs
        self.aff_nouns = aff_nouns

        # --- Dino-vit features --- #
        self.vit_feat_dim = 384
        self.stride_size = 14 #16
        self.patch_size = 14 #16

        # --- Residual MLP #
        #self.residual_MLP = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4), act_layer=nn.GELU, drop=0.3)
        
        # --- Affordance CAM generation --- #
        self.aff_conv_nouns = nn.Sequential(nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(self.vit_feat_dim), nn.ReLU(True), nn.Dropout(0.5),
                                           nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(self.vit_feat_dim), nn.ReLU(True),
                                           nn.Conv2d(self.vit_feat_dim, self.aff_nouns, 1))                                 
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.aff_fc_verbs = nn.Sequential(nn.Linear(self.vit_feat_dim, self.vit_feat_dim), nn.ReLU(True), nn.Dropout(0.5),
                                          nn.Linear(self.vit_feat_dim, self.aff_verbs))


    def forward(self, x_clss, x_patch, w_i, h_i):
        B = x_patch.shape[0]
        x_2D = x_patch.reshape(B, w_i // self.patch_size, h_i // self.patch_size, self.vit_feat_dim).permute(0, 3, 1, 2)
        
        nouns_cam = self.aff_conv_nouns(x_2D) # B x AFF_CLASSES x H x W
        nouns_logits = self.gap(nouns_cam).reshape(B, self.aff_nouns) # B x AFF_CLASSES
        
        verbs_logits = self.aff_fc_verbs(x_clss).reshape(B, self.aff_verbs) # B x AFF_CLASSES
        return {'1D_verb_prior': verbs_logits, '1D_noun_prior': nouns_logits, '2D_noun_prior': nouns_cam}