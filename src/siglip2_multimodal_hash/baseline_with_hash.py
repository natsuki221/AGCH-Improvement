# src/siglip2_multimodal_hash/baseline_with_hash.py
"""
SigLIP2-MLP Baseline 模型 + Hash Layer (AB-5)
用於公平對比 Hash 策略對 Baseline 的影響
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, GemmaTokenizerFast
from typing import Optional, Dict
from siglip2_multimodal_hash.model import HashLayer  # 復用既有的 HashLayer


class SigLIP2MLPHash(nn.Module):
    """
    Baseline + Hash 模型：SigLIP2 + MLP + Hash + 分類器

    架構：
    1. SigLIP2 Encoders -> [v_img, v_txt]
    2. Concat -> (B, 2D)
    3. MLP Projection -> (B, hidden_dim)
    4. Hash Layer -> (B, hash_bits)
    5. Classifier -> (B, num_classes)
    """

    def __init__(self, config):
        super().__init__()

        model_name = config.siglip2_variant

        # SigLIP2 encoders
        print(f"[Baseline+Hash] 載入 SigLIP2 模型: {model_name}")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.tokenizer = GemmaTokenizerFast.from_pretrained(model_name)
        self.siglip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ SigLIP2 載入成功")

        # Freeze towers
        if config.freeze_towers:
            for param in self.siglip_model.parameters():
                param.requires_grad = False
            print("✓ SigLIP2 towers frozen")

        # Embedding dim
        if hasattr(self.siglip_model.config, "projection_dim"):
            self.embed_dim = self.siglip_model.config.projection_dim
        elif hasattr(self.siglip_model.config, "text_config"):
            self.embed_dim = self.siglip_model.config.text_config.hidden_size
        else:
            self.embed_dim = 768

        # MLP Projection (降維/特徵轉換)
        input_dim = self.embed_dim * 2
        hidden_dim = config.get("baseline_hidden_dim", 512)
        dropout = config.get("baseline_dropout", 0.1)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Hash Layer
        # 注意：使用 hidden_dim 作為輸入
        hash_bits = config.hash.bits
        self.hash_layer = HashLayer(input_dim=hidden_dim, hash_bits=hash_bits, skip_hash=False)
        print(f"✓ Hash Layer initialized (bits={hash_bits})")

        # Classifier (接收 Hash Code)
        # 用於訓練時的分類 Loss
        self.classifier = nn.Linear(self.hash_layer.output_dim, config.classifier.num_classes)

        self.config = config

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        # Encode
        outputs = self.siglip_model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        v_img = outputs.image_embeds
        v_txt = outputs.text_embeds

        # 1. Concat
        fused = torch.cat([v_img, v_txt], dim=1)

        # 2. Project
        z = self.projector(fused)

        # 3. Hash
        h = self.hash_layer(z)

        # 4. Classify
        logits = self.classifier(h)

        # 為了相容 losses.py，需要返回 d_img, d_txt 等假資料
        # 因為 Baseline 沒有 decompose，所以 hash reg loss 正常計算，但 cosine loss 其實無意義
        # 我們將原始 embedding 當作 d_img, d_txt 傳回去，這樣 losses.py 會計算 cosine alignment
        # 這對 Baseline 來說也是合理的正則化 (希望 img 和 txt 原始向量接近)

        if return_components:
            return {
                "logits": logits,
                "h": h,
                "v_img": v_img,
                "v_txt": v_txt,
                # 用於 loss 計算的相容性欄位
                "d_img": v_img,
                "d_txt": v_txt,
                "z": z,
            }
        else:
            return logits

    @torch.no_grad()
    def get_hash(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """For inference: return hash codes"""
        outputs = self.siglip_model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        fused = torch.cat([outputs.image_embeds, outputs.text_embeds], dim=1)
        z = self.projector(fused)
        h = self.hash_layer(z)
        return h
