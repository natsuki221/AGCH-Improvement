# src/siglip2_multimodal_hash/__init__.py
"""
SigLIP2 Multimodal Hash 模組

對應手冊章節:
- §5 模型架構
- §11 實作細節

提供完整的多模態 Hash + KNN 分類系統
"""

from .model import DirectionMagnitudeDecomposer, HadamardFusion, HashLayer, MultimodalHashKNN

from .dataset import COCOMultiLabelDataset, create_dataloader

from .losses import hash_regularization, compute_total_loss, FocalLoss

from .utils import set_seed, get_gpu_memory_info, MemoryMonitor

from .knn import HashIndex, predict_tags, compute_knn_metrics

__all__ = [
    # 模型
    "DirectionMagnitudeDecomposer",
    "HadamardFusion",
    "HashLayer",
    "MultimodalHashKNN",
    # 資料集
    "COCOMultiLabelDataset",
    "create_dataloader",
    # 損失函數
    "hash_regularization",
    "compute_total_loss",
    "FocalLoss",
    # 工具
    "set_seed",
    "get_gpu_memory_info",
    "MemoryMonitor",
    # KNN
    "HashIndex",
    "predict_tags",
    "compute_knn_metrics",
]

__version__ = "1.0.0"
