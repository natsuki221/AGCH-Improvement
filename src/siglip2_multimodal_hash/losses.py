# src/siglip2_multimodal_hash/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def hash_regularization(
    h: torch.Tensor, lambda_balance: float = 0.1, lambda_decorr: float = 0.01
) -> torch.Tensor:
    """
    Hash 正則化損失（三項組合）

    Args:
        h: (batch_size, hash_bits) soft hash codes
        lambda_balance: bit balance 權重
        lambda_decorr: bit decorrelation 權重

    Returns:
        loss_hash: scalar tensor
    """
    # 1. Quantization loss（推向 ±1）
    loss_quant = torch.mean((torch.abs(h) - 1) ** 2)

    # 2. Bit balance loss（避免所有 bit 偏向同一極）
    bit_mean = torch.mean(h, dim=0)  # (hash_bits,)
    loss_balance = torch.mean(bit_mean**2)

    # 3. Bit decorrelation loss（鼓勵 bit 獨立）
    h_centered = h - torch.mean(h, dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / h.size(0)  # (B, B)
    loss_decorr = (torch.sum(cov**2) - torch.trace(cov**2)) / (h.size(1) ** 2)

    # 組合
    loss_hash = loss_quant + lambda_balance * loss_balance + lambda_decorr * loss_decorr

    return loss_hash


def compute_total_loss(outputs: dict, labels: torch.Tensor, config) -> tuple[torch.Tensor, dict]:
    """
    計算總損失

    Args:
        outputs: 模型輸出（包含 logits, h, d_img, d_txt 等）
        labels: (B, C) multi-hot labels
        config: 配置物件（已經是 loss 區塊）

    Returns:
        total_loss: 總損失
        loss_dict: 各項損失的字典
    """
    logits = outputs["logits"]
    h = outputs["h"]
    d_img = outputs["d_img"]
    d_txt = outputs["d_txt"]

    # 1. BCE Loss（主要監督訊號）
    loss_bce = F.binary_cross_entropy_with_logits(logits, labels)

    # 2. Cosine Alignment Loss
    loss_cos = 1 - F.cosine_similarity(d_img, d_txt, dim=1).mean()

    # 3. Hash Regularization
    loss_hash = hash_regularization(
        h,
        lambda_balance=config.hash_reg.lambda_balance,
        lambda_decorr=config.hash_reg.lambda_decorr,
    )

    # 組合總損失
    total_loss = (
        config.bce_weight * loss_bce
        + config.cosine_weight * loss_cos
        + config.hash_weight * loss_hash
    )

    # 返回損失字典（用於 logging）
    loss_dict = {
        "total": total_loss,
        "bce": loss_bce,
        "cos": loss_cos,
        "hash": loss_hash,
    }

    return loss_dict


class FocalLoss(nn.Module):
    """Focal Loss（處理類別不平衡，可選）"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw predictions
            labels: (B, C) multi-hot labels
        """
        probs = torch.sigmoid(logits)

        # 計算 focal weight
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # BCE loss with focal weight
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        focal_loss = self.alpha * focal_weight * bce

        return focal_loss.mean()
