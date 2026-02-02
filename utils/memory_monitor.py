# utils/memory_monitor.py
"""
記憶體監控工具

對應手冊章節:
- §9.5 記憶體管理工具

提供 GPU 與 CPU 記憶體即時監控功能，
支援警告閾值設定與峰值追蹤。
"""

import torch
import psutil
from typing import Dict, Optional


class MemoryMonitor:
    """記憶體監控工具"""

    def __init__(self, alert_threshold_gb: float = 14.5, total_vram_gb: float = 16.0):
        """
        Args:
            alert_threshold_gb: VRAM 使用警告閾值（GB）
            total_vram_gb: GPU 總 VRAM（GB），預設 16GB
        """
        self.alert_threshold_gb = alert_threshold_gb
        self.total_vram_gb = total_vram_gb
        self.peak_vram = 0

    def get_stats(self) -> Dict:
        """獲取完整記憶體統計"""
        stats = {}

        # GPU 記憶體
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9

            stats["gpu"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "free_gb": self.total_vram_gb - reserved,
                "utilization_%": allocated / self.total_vram_gb * 100,
            }

            # 更新峰值
            self.peak_vram = max(self.peak_vram, allocated)

            # 檢查是否超過閾值
            if allocated > self.alert_threshold_gb:
                stats["gpu"]["alert"] = True

        # CPU 記憶體
        ram = psutil.virtual_memory()
        stats["cpu"] = {
            "used_gb": ram.used / 1e9,
            "available_gb": ram.available / 1e9,
            "total_gb": ram.total / 1e9,
            "percent": ram.percent,
        }

        return stats

    def print_stats(self, prefix: str = ""):
        """列印記憶體統計"""
        stats = self.get_stats()

        if "gpu" in stats:
            gpu = stats["gpu"]
            print(
                f"{prefix}GPU: {gpu['allocated_gb']:.2f}GB / {self.total_vram_gb}GB "
                f"({gpu['utilization_%']:.1f}%), "
                f"Peak: {self.peak_vram:.2f}GB"
            )

            if gpu.get("alert"):
                print(f"  ⚠️  WARNING: VRAM usage high!")

        cpu = stats["cpu"]
        print(
            f"{prefix}RAM: {cpu['used_gb']:.1f}GB / {cpu['total_gb']:.1f}GB "
            f"({cpu['percent']:.1f}%)"
        )

    def reset_peak(self):
        """重置峰值統計"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_vram = 0

    def check_alert(self) -> bool:
        """檢查是否超過警告閾值"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            return allocated > self.alert_threshold_gb
        return False

    def get_summary(self) -> str:
        """獲取簡短的記憶體摘要字串"""
        stats = self.get_stats()
        if "gpu" in stats:
            gpu = stats["gpu"]
            return f"GPU: {gpu['allocated_gb']:.1f}GB | Peak: {self.peak_vram:.1f}GB"
        return "GPU: N/A"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    獲取 GPU 記憶體使用資訊（獨立函數版本）

    Returns:
        包含記憶體資訊的字典
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "free_gb": 16.0 - reserved,
        }
    return {}


# =========================================
# 使用範例
# =========================================
if __name__ == "__main__":
    # 建立監控實例
    monitor = MemoryMonitor(alert_threshold_gb=14.5)

    # 列印當前狀態
    print("現在的記憶體使用狀態:")
    monitor.print_stats("  ")

    # 取得統計資料
    stats = monitor.get_stats()
    print(f"\n詳細統計: {stats}")
