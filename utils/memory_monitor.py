# utils/memory_monitor.py

import torch
import psutil
import GPUtil


class MemoryMonitor:
    """記憶體監控工具"""

    def __init__(self, alert_threshold_gb=14.5):
        self.alert_threshold_gb = alert_threshold_gb
        self.peak_vram = 0

    def get_stats(self):
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
                "free_gb": 16.0 - reserved,
                "utilization_%": allocated / 16.0 * 100,
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
            "percent": ram.percent,
        }

        return stats

    def print_stats(self, prefix=""):
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

        cpu = stats["cpu"]
        print(f"{prefix}RAM: {cpu['used_gb']:.1f}GB / {42:.1f}GB " f"({cpu['percent']:.1f}%)")

    def reset_peak(self):
        """重置峰值統計"""
        torch.cuda.reset_peak_memory_stats()
        self.peak_vram = 0


# 使用範例
monitor = MemoryMonitor(alert_threshold_gb=14.5)

# 訓練前
monitor.print_stats("訓練前 - ")

# 訓練中（定期檢查）
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... 訓練程式碼 ...

        if batch_idx % 100 == 0:
            monitor.print_stats(f"Epoch {epoch}, Batch {batch_idx} - ")
