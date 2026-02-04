# src/siglip2_multimodal_hash/utils.py

import torch
import random
import numpy as np
from typing import Dict


def set_seed(seed: int = 42):
    """設定隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 以下設定會減慢訓練，但保證可重現
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_gpu_memory_info() -> Dict[str, float]:
    """獲取 GPU 記憶體使用資訊"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "total_gb": total,
            "free_gb": total - reserved,
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0, "total_gb": 0, "free_gb": 0}


class MemoryMonitor:
    """記憶體監控工具"""

    def __init__(self, alert_threshold_gb: float = 14.5):
        self.alert_threshold_gb = alert_threshold_gb
        self.peak_vram = 0

    def get_stats(self) -> dict:
        """獲取完整記憶體統計"""
        stats = {}

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9

            stats["gpu"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "free_gb": 16.0 - reserved,
                "utilization_%": allocated / 16.0 * 100,
            }

            self.peak_vram = max(self.peak_vram, allocated)

            if allocated > self.alert_threshold_gb:
                stats["gpu"]["alert"] = True

        return stats

    def print_stats(self, prefix: str = ""):
        """列印記憶體統計"""
        stats = self.get_stats()

        if "gpu" in stats:
            gpu = stats["gpu"]
            print(
                f"{prefix}GPU: {gpu['allocated_gb']:.2f}GB / 16GB "
                f"({gpu['utilization_%']:.1f}%), "
                f"Peak: {self.peak_vram:.2f}GB"
            )

            if gpu.get("alert"):
                print(f"  ⚠️  WARNING: VRAM usage high!")

    def reset_peak(self):
        """重置峰值統計"""
        torch.cuda.reset_peak_memory_stats()
        self.peak_vram = 0
