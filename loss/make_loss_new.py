# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch
import torch
import torch.nn as nn

import torch
import torch.nn as nn

def get_lambda(epoch, T_total=60, T_warm=10, lambda_max=0.5):
    if epoch < T_warm:
        return 0.0
    else:
        return lambda_max * (epoch - T_warm) / (T_total - T_warm)

# ===== Reward 计算函数 =====
def compute_reward(pred_logits, gt_labels, base_probs, values):
    """
    pred_logits: [B, 4, C] 每个 localfeat 的分类 logit
    gt_labels:   [B]       真实类别标签
    base_probs:  [B, 4]    每个 localfeat 单独分类的概率
    weighted_probs: [B, 4] 融合后分类的概率（分配回 localfeat）

    返回:
      R: [B, 4]  每个 localfeat 的 reward
    """
    pred_logits=torch.stack(pred_logits[1:],dim=1)
    base_probs=torch.stack(base_probs,dim=1)
    B, N, C =pred_logits.shape

    # 分类预测结果
    pred_labels = pred_logits.argmax(dim=-1)  # [B, 4]
   # import pdb
   # pdb.set_trace()
    # R_cls: 分类正确奖励
    gt_expand = gt_labels.unsqueeze(1).expand(B, N)  # [B,4]
    R_cls = (pred_labels == gt_expand).float()  # [B,4]

    weighted_gt = torch.gather(pred_logits, 2, gt_expand.unsqueeze(-1)).squeeze(-1)  # shape (B,4)
    base_gt = torch.gather(base_probs, 2, gt_expand.unsqueeze(-1)).squeeze(-1)        # shape (B,4)

# 比较概率生成 Rp
    R_p = (weighted_gt > base_gt).float()  # shape (B,4)，满足条件为1，否则为0

    # R_p: 加权前后置信度比较
    #R_p = torch.where(weighted_probs > base_probs, 
    #                  torch.zeros_like(base_probs), 
    #                  torch.ones_like(base_probs))  # [B,4]

    # 总 reward
    R = R_cls + R_p
    reward_loss = F.mse_loss(values, R)
    return reward_loss
def compute_multi_prediction_mse(scores, label, weights, mode="mean", loss_weights=None):
    """
    支持多个分类分数的 MSE 计算，并根据 mode 聚合 loss

    Args:
        scores: list[Tensor] 或 Tensor，每个 shape = [B, C]
        label: [B] 真实标签
        weights: [B, 1] 或 [B] 权重
        mode: {"mean", "sum", "weighted"} 聚合方式
        loss_weights: list[float]，当 mode="weighted" 时必须提供，对应每个分支的权重

    Returns:
        total_loss: 标量张量，总损失
        losses: list[Tensor]，每个分支的损失
        results: list[Tensor]，每个分支的 0/1 预测正确性 [B,1]
    """
    if isinstance(scores, torch.Tensor):  # 兼容单个输入
        scores = [scores]

    # 保证 weights shape = [B,1]
    if weights.dim() == 1:
        weights = weights.unsqueeze(1)

    mse_loss = nn.MSELoss()
    losses, results = [], []
    i=0
    for cls_score in scores:
        pred = torch.argmax(cls_score, dim=1)         # [B]
        result = (pred == label).float().unsqueeze(1) # [B,1]
        loss = mse_loss(weights[:,i], result)
        i+=1

        losses.append(loss)
        results.append(result)

    # 聚合总 loss
    if mode == "mean":
        total_loss = torch.stack(losses).mean()
    elif mode == "sum":
        total_loss = torch.stack(losses).sum()
    elif mode == "weighted":
        assert loss_weights is not None, "loss_weights must be provided when mode='weighted'"
        assert len(loss_weights) == len(losses), "loss_weights length must match number of scores"
        total_loss = sum(w * l for w, l in zip(loss_weights, losses))
    else:
        raise ValueError("mode must be 'mean', 'sum' or 'weighted'")

    #return total_loss, losses, results
    return total_loss

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        #def loss_func(score, feat, target, target_cam, i2tscore = None):
#        def loss_func(score,oriscore, feat, target, target_cam,cls_score_view,view_feature,value, i2tscore = None):
        def loss_func(score, oriscore,feat, target, target_cam,cls_score_view,view_feature,value,epoch,i2tscore = None):
#            import pdb
#            pdb.set_trace()
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
#                    view_LOSS = xent(cls_score_view, target_cam)
                    #import pdb
                    #pdb.set_trace()
                    orthogonal_loss=torch.cosine_similarity(feat[1], view_feature).abs().mean()
                    #view_loss=cfg.MODEL.ID_LOSS_WEIGHT*view_LOSS+0.1*orthogonal_loss
 #                   view_loss=cfg.MODEL.ID_LOSS_WEIGHT*view_LOSS
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                  #  loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                   #only cls reward************************** 
                   #rl_loss=compute_multi_prediction_mse(score[2:],target,weights)	
                   #cls+prob  reward************************** 
                    rl_loss=compute_reward(score,target,oriscore, value)	
                    lambda_rl=get_lambda(epoch=epoch)
                    loss=loss+lambda_rl*rl_loss

#                    loss=loss+0.1*rl_loss
                    if i2tscore != None:
                        #I2TLOSS = [xent(i2t, target) for i2t in i2tscore[0:]]
                        I2TLOSS = xent(i2tscore, target) 
                        #I2TLOSS = sum(I2TLOSS)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


