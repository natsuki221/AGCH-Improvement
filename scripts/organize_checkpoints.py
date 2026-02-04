#!/usr/bin/env python3
"""
整理 Checkpoint 檔案
將 experiments/checkpoints/ 下的檔案移動到 outputs/checkpoints/{experiment_name}/
並挑選每個 fold 最佳的模型改名為 best_model.pth
"""

import glob
import torch
import shutil
from pathlib import Path
import re
import os


def main():
    source_dir = Path("experiments/checkpoints")
    base_target_dir = Path("outputs/checkpoints")

    files = glob.glob(str(source_dir / "*.pth"))
    print(f"找到 {len(files)} 個 .pth 檔案")

    # 統計每個 fold 的最佳模型
    best_models = {}  # fold_name -> (mAP, file_path)

    for f in sorted(files):
        try:
            # 完整讀取會比較慢，但我需要 config 中的 experiment.name
            # 為了加速，也可以嘗試從 map_location='cpu' 讀取
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)

            config = checkpoint.get("config", {})
            exp_name = config.get("experiment", {}).get("name")
            val_mAP = checkpoint.get("val_mAP", 0.0)
            epoch = checkpoint.get("epoch", -1)

            if not exp_name:
                print(f"⚠️  跳過無法識別實驗名稱的檔案: {Path(f).name}")
                continue

            # 建立目標目錄
            target_dir = base_target_dir / exp_name
            target_dir.mkdir(parents=True, exist_ok=True)

            file_path = Path(f)
            target_path = target_dir / file_path.name

            # 移動檔案
            print(f"📦 移動: {file_path.name} -> {target_dir}/")
            shutil.move(str(file_path), str(target_path))

            # 追蹤最佳模型
            if exp_name not in best_models or val_mAP > best_models[exp_name][0]:
                best_models[exp_name] = (val_mAP, target_path, epoch)

        except Exception as e:
            print(f"❌ 處理錯誤 {f}: {e}")

    # 為每個 fold 建立 best_model.pth 連結或副本
    print("\n🔗 建立最佳模型連結...")
    for exp_name, (mAP, path, epoch) in best_models.items():
        target_dir = base_target_dir / exp_name
        link_path = target_dir / "best_model.pth"

        # 移除舊的連結或檔案
        if link_path.exists():
            link_path.unlink()

        # 複製檔案（因為可能不想破壞原始按照 epoch 命名的檔案結構）
        shutil.copy2(path, link_path)
        print(f"✓ {exp_name}: 最佳 mAP={mAP:.4f} (Epoch {epoch}) -> {link_path}")

    print("\n✅ 整理完成！")


if __name__ == "__main__":
    main()
