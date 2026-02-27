import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPositiveLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, image_feats, id_labels, plat_labels, text_feats):
        """
        image_feats: [B, D]
        id_labels:   [B] (ID label)
        plat_labels: [B] (platform label, 这里其实不用，除非要做 hard negative)
        text_feats:  [num_classes, num_platform, D]
        """
        B, D = image_feats.shape
        num_classes, num_platform, _ = text_feats.shape

        # 归一化
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats  = F.normalize(text_feats, dim=-1)

        # flatten text feats: [num_classes*num_platform, D]
        text_feats_flat = text_feats.view(-1, D)

        # similarity logits: [B, num_classes*num_platform]
        logits = (image_feats @ text_feats_flat.T) / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)

        # 构造 positive indices
        device = id_labels.device
        pos_index = id_labels.unsqueeze(1) * num_platform + torch.arange(num_platform, device=device).unsqueeze(0)
        # pos_index: [B, num_platform]

        # 方法 A: 正样本 log-prob 平均
        pos_log_probs = log_probs.gather(1, pos_index)  # [B, num_platform]
        loss = -(pos_log_probs.mean(dim=1))  # [B]

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
