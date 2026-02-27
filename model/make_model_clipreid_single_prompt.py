import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from collections import OrderedDict
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticLocalAndGlobalV(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256):
        super().__init__()
        self.weight_fc = nn.Linear(feat_dim, 1) # score
        self.local_fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: [B, 4, D]
        """
        B, N, D = x.shape
      #  N=len(x)
        assert N == 4

        # 1) score
        scores = self.weight_fc(x).squeeze(-1)  # [B, 4]
        weights = F.softmax(scores, dim=1)      # [B, 4]

        # 2) 加权特征
        x_tilde = x * weights.unsqueeze(-1)     # [B,4,D]

        # 3) local V
        local_V = self.local_fc(x_tilde).squeeze(-1)  # [B, 4]

        # 4) global V = 加权融合 local_V
        global_V = torch.sum(local_V * weights, dim=1, keepdim=True)  # [B,1]

       # return local_V, global_V, weights
        return local_V, x_tilde, weights

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
#        import pdb
#        pdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.classifier_view = nn.Linear(self.in_planes, self.camera_num, bias=False)
        self.classifier_view.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_view = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_view.bias.requires_grad_(False)
        self.bottleneck_view.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        

        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)



        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        #self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding,clip_model)
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.attention = nn.Sequential(
            nn.Linear(self.in_planes, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.value=CriticLocalAndGlobalV()
    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None,img_features = None):
        if get_text ==True  and get_image==True:
            #prompts = self.prompt_learner(label,img_features) 
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            #import pdb
            #pdb.set_trace()
            image_features_last, image_features, image_features_proj,view_feat,localfeat= self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj,view_feature,ori_localfeat = self.image_encoder(x, cv_embed) 
            #image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
# *********************************************************ori weights calculate
#        attn_weights = torch.softmax(self.attention(torch.stack(localfeat,dim=1)), dim=1)
#        localfeat = torch.stack(localfeat,dim=1)* attn_weights
# *********************************************************ori weights calculate
        #localfeat = torch.stack(localfeat,dim=1)
        local_v,localfeat,_=self.value(torch.stack(ori_localfeat,dim=1))
        feat = self.bottleneck(img_feature) 
        view_feat = self.bottleneck_view(view_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        local_feat_1_bn = self.bottleneck_1(localfeat[:,0,:])
        local_feat_2_bn = self.bottleneck_2(localfeat[:,1,:])
        local_feat_3_bn = self.bottleneck_3(localfeat[:,2,:])
        local_feat_4_bn = self.bottleneck_4(localfeat[:,3,:])
        ori_local_feat_1_bn = self.bottleneck_1(ori_localfeat[0])
        ori_local_feat_2_bn = self.bottleneck_2(ori_localfeat[1])
        ori_local_feat_3_bn = self.bottleneck_3(ori_localfeat[2])
        ori_local_feat_4_bn = self.bottleneck_4(ori_localfeat[3])
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_view = self.classifier_view(view_feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            cls_score_1 = self.classifier_1(local_feat_1_bn)
            cls_score_2 = self.classifier_2(local_feat_2_bn)
            cls_score_3 = self.classifier_3(local_feat_3_bn)
            cls_score_4 = self.classifier_4(local_feat_4_bn)
            
            ori_cls_score_1 = self.classifier_1(ori_local_feat_1_bn)
            ori_cls_score_2 = self.classifier_2(ori_local_feat_2_bn)
            ori_cls_score_3 = self.classifier_3(ori_local_feat_3_bn)
            ori_cls_score_4 = self.classifier_4(ori_local_feat_4_bn)
            #return [cls_score, cls_score_proj,cls_score_1,cls_score_2,cls_score_3,cls_score_4], [img_feature_last, img_feature, img_feature_proj,localfeat[0],localfeat[1],localfeat[2],localfeat[3]], img_feature_proj,cls_score_view,view_feature
            #return [cls_score, cls_score_proj,cls_score_1,cls_score_2,cls_score_3,cls_score_4], [img_feature_last, img_feature, img_feature_proj,localfeat[:,0,:],localfeat[:,1,:],localfeat[:,2,:],localfeat[:,3,:]], img_feature_proj,cls_score_view,view_feature,attn_weights
#            return [cls_score, cls_score_proj,cls_score_1,cls_score_2,cls_score_4],[ori_cls_score_1,ori_cls_score_2,ori_cls_score_3,ori_cls_score_4], [img_feature_last, img_feature, img_feature_proj,localfeat[:,0,:],localfeat[:,1,:],localfeat[:,3,:]], img_feature_proj,cls_score_view,view_feature,local_v
            return [cls_score, cls_score_proj,cls_score_1,cls_score_2,cls_score_4],[ori_cls_score_1,ori_cls_score_2,ori_cls_score_3,ori_cls_score_4], [img_feature_last, img_feature, img_feature_proj,localfeat[:,0,:],localfeat[:,1,:],localfeat[:,2,:],localfeat[:,3,:]], img_feature_proj,cls_score_view,view_feature,local_v
#            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                #return torch.cat([feat, feat_proj,local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
                return torch.cat([feat, feat_proj,local_feat_1_bn / 3, local_feat_2_bn / 3,  local_feat_4_bn / 3], dim=1)
            else:
#                return torch.cat([img_feature, img_feature_proj], dim=1)
            #    return torch.cat([img_feature, img_feature_proj,localfeat[:,0,:]/4,localfeat[:,1,:]/4,localfeat[2]/4,localfeat[3]/4], dim=1)
                return torch.cat([img_feature, img_feature_proj,localfeat[:,0,:],localfeat[:,1,:],localfeat[:,2,:],localfeat[:,3,:]], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    print('**********')
    print(f'url is {url}')
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            #ctx_init = "A photo of a X X X X person.Bridging Aerial and Ground Views."
            ctx_init = "A photo of a X X X X person."
            ctx_init1 = "A photo of a X X X X platform."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        ctx_init1 = ctx_init1.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        tokenized_prompts1 = clip.tokenize(ctx_init1).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
            embedding1 = token_embedding(tokenized_prompts1).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.tokenized_prompts1 = tokenized_prompts1  # torch.Tensor
        self.embedding = embedding
        self.embedding1 = embedding1
        n_cls_ctx = 4
        num_plats = 3
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        plat_vectors = torch.empty(num_plats, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        nn.init.normal_(plat_vectors, std=0.02)
        self.plat_ctx = nn.Parameter(plat_vectors) 
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.register_buffer("token_prefix1", embedding1[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix1", embedding1[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    #def forward(self, label,target_cam):
    def forward(self, label):
        #import pdb
        #pdb.set_trace()
        cls_ctx = self.cls_ctx[label] 
        plat_ctx = self.plat_ctx[0] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prefix1 = self.token_prefix1.expand(b, -1, -1) 
        suffix1 = self.token_suffix1.expand(b, -1, -1) 
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

#        prompts1 = torch.cat(
#            [
#                prefix1,  # (n_cls, 1, dim)
#                plat_ctx,     # (n_cls, n_ctx, dim)
#                suffix1,  # (n_cls, *, dim)
#            ],
#            dim=1,
#        ) 
        
        return prompts 
#
#class PromptLearner(nn.Module):
#    def __init__(self, num_class, dataset_name, dtype, token_embedding,clip_model):
#        super().__init__()
#        if dataset_name == "VehicleID" or dataset_name == "veri":
#            ctx_init = "A photo of a X X X X vehicle."
#        else:
#            ctx_init = "A photo of a X X X X person."
#
#        ctx_dim = 512
#        # use given words to initialize context vectors
#        ctx_init = ctx_init.replace("_", " ")
#        n_ctx = 4
#
#        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
#        with torch.no_grad():
#            embedding = token_embedding(tokenized_prompts).type(dtype) 
#        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#        n_cls_ctx = 4
#        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
#        nn.init.normal_(cls_vectors, std=0.02)
#        self.cls_ctx = nn.Parameter(cls_vectors)
#        vis_dim = clip_model.visual.output_dim
#        self.meta_net = nn.Sequential(
#            OrderedDict(
#                [
#                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#                    ("relu", nn.ReLU(inplace=True)),
#                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
#                ]
#            )
#        )
#        
#        # These token vectors will be saved when in save_model(),
#        # but they should be ignored in load_model() as we want to use
#        # those computed using the current class names
#        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
#        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
#        self.num_class = num_class
#        self.n_cls_ctx = n_cls_ctx
#
#    def forward(self, label, image_features):
#        b = label.shape[0]
#        ctx = self.cls_ctx[label]  # (batch_size, n_ctx, ctx_dim)
#        # print(f"ctx -- {ctx.shape}")
#        bias = self.meta_net(image_features)  # (batch_size, ctx_dim)
#        # print(f"bias -- {bias.shape}")
#        bias = bias.unsqueeze(1)  # (batch_size, 1, ctx_dim)
#        # print(f"bias -- {bias.shape}")
#        ctx_shifted = ctx + bias  # (batch_size, n_ctx, ctx_dim)  (1, 4, 512)
#        # print(f"The shape of the ctx_shifted: {ctx_shifted.shape}\n")
#
#        prefix = self.token_prefix.expand(1, -1, -1)  # (batch_size, prefix_length, ctx_dim)  (1, 5, 512)
#        suffix = self.token_suffix.expand(1, -1, -1)  # (batch_size, suffix_length, ctx_dim)  (1, 68, 512)
#        prompts = []
#        for ctx_shifted_i in ctx_shifted:
#            ctx_i = ctx_shifted_i.unsqueeze(0)  # (1, n_ctx, ctx_dim)  (1, 4, 512)
#            # ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_id, -1, -1)  # (n_id, n_ctx, ctx_dim)  (*, 4, 512)
#            # print(f"The shape of the prefix: {prefix.shape}")
#            # print(f"The shape of the ctx_i: {ctx_shifted_i.shape}")
#            # print(f"The shape of the suffix: {suffix.shape}\n")
#
#            prompts_i = torch.cat([prefix, ctx_i, suffix], dim=1)  # (n_id, fixed_text_length, ctx_dim)  (*, 77, 512)
#            prompts.append(prompts_i)
#
#        prompts = torch.stack(prompts)
#        prompts = prompts.squeeze(dim=1)
#
#        return prompts
