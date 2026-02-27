import torch
import torch.nn as nn
import torch.nn.functional as F

class PlatformAwareLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, image_feats, id_labels, plat_labels, text_feats):
        """
        image_feats: [B, D]
        id_labels:   [B]
        plat_labels: [B]
        text_feats:  [num_classes, num_platform, D]
        """
        B, D = image_feats.shape
        num_classes, num_platform, _ = text_feats.shape

        # normalize
        image_feats = F.normalize(image_feats, dim=-1)  # [B, D]
        text_feats = F.normalize(text_feats, dim=-1)    # [num_classes, num_platform, D]

        # 取出对应ID的所有platform文本特征: [B, num_platform, D]
        text_feats_id = text_feats[id_labels]

        # 相似度计算: [B, num_platform]
        logits = torch.einsum("bd,bpd->bp", image_feats, text_feats_id)
        logits = logits / self.temperature

        # 构造mask，排除同platform之外的正负关系
        device = image_feats.device
        mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, num_platform]
        mask[torch.arange(B, device=device), plat_labels] = 1

        # 取正样本 logit: [B, 1]
        pos_logits = logits[mask].unsqueeze(1)

        # 拼接 (正 + 负) -> softmax
        # [B, num_platform]
        all_logits = logits

        log_probs = F.log_softmax(all_logits, dim=1)  # [B, num_platform]

        # loss = -log P(pos)
        loss = -log_probs[mask]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
