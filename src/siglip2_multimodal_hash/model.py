# src/siglip2_multimodal_hash/model.py

import torch
import torch.nn as nn
from transformers import Siglip2Model, Siglip2Processor
from typing import Optional, Dict


class DirectionMagnitudeDecomposer(nn.Module):
    """方向/幅度分解模組"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v: (batch_size, dim) raw embedding
        Returns:
            direction: (batch_size, dim) unit vector
            magnitude: (batch_size, 1) log-norm
        """
        norm = torch.norm(v, p=2, dim=1, keepdim=True)  # (B, 1)
        direction = v / (norm + self.eps)  # (B, D)
        magnitude = torch.log(norm + self.eps)  # (B, 1)
        return direction, magnitude


class HadamardFusion(nn.Module):
    """Hadamard 乘積融合模組"""

    def __init__(
        self, embed_dim: int, mlp_dims: list[int], dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()

        # Input: [d_img, d_txt, p_dir, m_img, m_txt]
        input_dim = embed_dim * 3 + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in mlp_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU() if activation == "relu" else nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, d_img: torch.Tensor, d_txt: torch.Tensor, m_img: torch.Tensor, m_txt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            d_img: (B, D) image direction
            d_txt: (B, D) text direction
            m_img: (B, 1) image magnitude
            m_txt: (B, 1) text magnitude
        Returns:
            z: (B, mlp_dims[-1]) fused embedding
        """
        p_dir = d_img * d_txt  # Hadamard product
        x = torch.cat([d_img, d_txt, p_dir, m_img, m_txt], dim=1)
        z = self.mlp(x)
        return z


class HashLayer(nn.Module):
    """Hash 層"""

    def __init__(self, input_dim: int, hash_bits: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hash_bits)
        self.hash_bits = hash_bits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns soft hash codes in [-1, 1]"""
        h = torch.tanh(self.fc(z))
        return h

    def binarize(self, h: torch.Tensor) -> torch.Tensor:
        """For inference: convert to hard binary {-1, 1}"""
        return torch.sign(h)


class MultimodalHashKNN(nn.Module):
    """完整模型：SigLIP2 + 方向/幅度分解 + Hadamard 融合 + Hash + KNN"""

    def __init__(self, config):
        super().__init__()

        # SigLIP2 encoders
        print(f"載入 SigLIP2 模型: {config.model.siglip2_variant}")
        self.processor = Siglip2Processor.from_pretrained(config.model.siglip2_variant)
        self.model = Siglip2Model.from_pretrained(config.model.siglip2_variant)

        # ⚠️ 必須凍結 towers（RTX 5080 16GB 限制）
        if config.model.freeze_towers:
            for param in self.model.parameters():
                param.requires_grad = False
            print("✓ SigLIP2 towers frozen (saving ~7.5GB VRAM)")

        # 獲取 embedding 維度
        self.embed_dim = self.model.config.projection_dim  # 768 for base

        # Decomposer
        self.decomposer = DirectionMagnitudeDecomposer(eps=config.model.decomposer.eps)

        # Fusion
        self.fusion = HadamardFusion(
            embed_dim=self.embed_dim,
            mlp_dims=config.model.fusion.mlp_dims,
            dropout=config.model.fusion.dropout,
            activation=config.model.fusion.activation,
        )

        # Hash layer
        self.hash_layer = HashLayer(
            input_dim=config.model.fusion.mlp_dims[-1], hash_bits=config.model.hash.bits
        )

        # Classifier head (for training)
        self.classifier = nn.Linear(
            config.model.hash.bits,
            config.model.classifier.num_classes,
            bias=config.model.classifier.use_bias,
        )

        self.config = config

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
        outputs = self.model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        v_img = outputs.image_embeds  # (B, D)
        v_txt = outputs.text_embeds  # (B, D)

        # Decompose
        d_img, m_img = self.decomposer(v_img)
        d_txt, m_txt = self.decomposer(v_txt)

        # Fuse
        z = self.fusion(d_img, d_txt, m_img, m_txt)

        # Hash
        h = self.hash_layer(z)

        # Classify
        logits = self.classifier(h)

        if return_components:
            return {
                "logits": logits,
                "h": h,
                "d_img": d_img,
                "d_txt": d_txt,
                "m_img": m_img,
                "m_txt": m_txt,
                "z": z,
                "v_img": v_img,
                "v_txt": v_txt,
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
        outputs = self.model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )
        v_img = outputs.image_embeds
        v_txt = outputs.text_embeds
        d_img, m_img = self.decomposer(v_img)
        d_txt, m_txt = self.decomposer(v_txt)
        z = self.fusion(d_img, d_txt, m_img, m_txt)
        h = self.hash_layer(z)
        return h
