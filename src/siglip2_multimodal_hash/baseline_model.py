# src/siglip2_multimodal_hash/baseline_model.py
"""
SigLIP2-MLP Baseline 模型
直接使用 MLP 分類器，無 decomposition、無 hash、無 KNN
用於對比驗證改進方法的效果
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor, GemmaTokenizerFast
from typing import Optional, Dict


class SigLIP2MLPBaseline(nn.Module):
    """
    Baseline 模型：SigLIP2 + MLP 分類器

    與改進版本 (MultimodalHashKNN) 的差異：
    - ❌ 無方向/幅度分解 (DirectionMagnitudeDecomposer)
    - ❌ 無 Hadamard 融合
    - ❌ 無 Hash 層
    - ❌ 無 KNN 推論
    - ✅ 直接 concat [v_img, v_txt] 並用 MLP 分類
    """

    def __init__(self, config):
        super().__init__()

        model_name = config.siglip2_variant

        # SigLIP2 encoders
        print(f"[Baseline] 載入 SigLIP2 模型: {model_name}")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
        self.tokenizer = GemmaTokenizerFast.from_pretrained(model_name)
        self.siglip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ SigLIP2 載入成功 (Model: {type(self.siglip_model).__name__})")

        # ⚠️ 必須凍結 towers（RTX 5080 16GB 限制）
        if config.freeze_towers:
            for param in self.siglip_model.parameters():
                param.requires_grad = False
            print("✓ SigLIP2 towers frozen (saving ~7.5GB VRAM)")

        # 獲取 embedding 維度
        if hasattr(self.siglip_model.config, "projection_dim"):
            self.embed_dim = self.siglip_model.config.projection_dim
        elif hasattr(self.siglip_model.config, "text_config"):
            self.embed_dim = self.siglip_model.config.text_config.hidden_size
        else:
            self.embed_dim = 768  # fallback for siglip2-base
        print(f"  Embedding dim: {self.embed_dim}")

        # MLP 分類器 (簡單 concat 後分類)
        # 輸入: [v_img, v_txt] = 2 * embed_dim
        input_dim = self.embed_dim * 2
        hidden_dim = config.get("baseline_hidden_dim", 512)
        dropout = config.get("baseline_dropout", 0.1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, config.classifier.num_classes),
        )

        self.config = config
        print(
            f"✓ Baseline MLP 分類器建立完成 ({input_dim} -> {hidden_dim} -> {config.classifier.num_classes})"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, C, H, W) images
            input_ids: (B, L) text tokens
            attention_mask: (B, L) attention mask
            return_components: whether to return intermediate results

        Returns:
            logits or dict of components
        """
        # Encode
        outputs = self.siglip_model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        v_img = outputs.image_embeds  # (B, D)
        v_txt = outputs.text_embeds  # (B, D)

        # 直接 concat（無 decomposition、無 Hadamard）
        fused = torch.cat([v_img, v_txt], dim=1)  # (B, 2D)

        # 分類
        logits = self.classifier(fused)

        if return_components:
            return {
                "logits": logits,
                "v_img": v_img,
                "v_txt": v_txt,
                "fused": fused,
            }
        else:
            return logits
