import torch
import torch.nn.functional as F

def symmetric_clip_loss_batch_pairing(image_feats, text_feats_batch, temperature=0.07, reduction='mean'):
    """
    image_feats: [B, D]
    text_feats_batch: [B, D]    # 每个样本对应的 joint 文本向量（由 id, platform 确定）
    targets: implicit as indices 0..B-1
    """
    B, D = image_feats.shape
    assert text_feats_batch.shape == (B, D)

    # Normalize
    image_norm = F.normalize(image_feats, dim=-1)    # [B, D]
    text_norm  = F.normalize(text_feats_batch, dim=-1)  # [B, D]

    logits = (image_norm @ text_norm.t()) / temperature  # [B, B]

    targets = torch.arange(B, device=image_feats.device)

    # i->t (rows: images, cols: texts)
    loss_i2t = F.cross_entropy(logits, targets, reduction=reduction)
    # t->i (rows: texts, cols: images) => logits.T
    loss_t2i = F.cross_entropy(logits.t(), targets, reduction=reduction)

    return 0.5 * (loss_i2t + loss_t2i)

