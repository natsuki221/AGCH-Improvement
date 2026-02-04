#!/usr/bin/env python3
"""
臨時腳本 - 分析 experiments/checkpoints/ 下的現有 checkpoint
註：這些 checkpoint 來自混合的 5-fold 訓練結果
"""

import glob
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import re


def main():
    checkpoint_dir = Path("experiments/checkpoints")
    pattern = str(checkpoint_dir / "*.pth")
    files = glob.glob(pattern)

    print("=" * 60)
    print("📊 分析現有 Checkpoint 結果")
    print("=" * 60)
    print(f"搜尋路徑: {pattern}")
    print(f"找到 {len(files)} 個 checkpoint 檔案\n")

    if not files:
        print("⚠️ 未找到任何 checkpoint 檔案")
        return

    # 按時間分組分析（每個 fold 訓練的 checkpoint 時間接近）
    results = []
    for f in sorted(files):
        try:
            # 只讀取必要的資訊（不載入完整模型以節省記憶體）
            ckpt = torch.load(f, map_location="cpu", weights_only=False)

            val_mAP = ckpt.get("val_mAP", None)
            epoch = ckpt.get("epoch", -1)
            config = ckpt.get("config", {})
            exp_name = config.get("experiment", {}).get("name", "unknown")

            # 從檔名提取 mAP（作為備份）
            match = re.search(r"mAP([\d.]+)", Path(f).name)
            if val_mAP is None and match:
                val_mAP = float(match.group(1))

            if val_mAP is not None:
                results.append(
                    {
                        "file": Path(f).name,
                        "epoch": epoch,
                        "mAP": val_mAP,
                        "exp_name": exp_name,
                        "mtime": Path(f).stat().st_mtime,
                    }
                )
        except Exception as e:
            print(f"❌ 讀取錯誤 {f}: {e}")

    if not results:
        print("⚠️ 未找到有效的 mAP 結果")
        return

    # 按實驗名稱分組
    by_exp = defaultdict(list)
    for r in results:
        by_exp[r["exp_name"]].append(r)

    print(f"📋 找到 {len(by_exp)} 個實驗:\n")

    all_best_maps = []

    for exp_name, exp_results in sorted(by_exp.items()):
        # 取該實驗的最佳 mAP
        best = max(exp_results, key=lambda x: x["mAP"])
        all_best_maps.append(best["mAP"])

        print(f"  {exp_name}:")
        print(f"    - 最佳 mAP: {best['mAP']:.4f} (Epoch {best['epoch']})")
        print(f"    - Checkpoint: {best['file']}")
        print()

    # 如果有多個實驗（fold），計算統計量
    if len(all_best_maps) >= 2:
        print("=" * 60)
        print("🏆 跨實驗統計 (各實驗最佳 mAP)")
        print("=" * 60)
        print(f"  Mean:   {np.mean(all_best_maps):.4f}")
        print(f"  Std:    {np.std(all_best_maps, ddof=1):.4f}")
        print(f"  Min:    {np.min(all_best_maps):.4f}")
        print(f"  Max:    {np.max(all_best_maps):.4f}")
        print(f"  Median: {np.median(all_best_maps):.4f}")
        print()
        print(
            f"📝 論文格式: mAP = {np.mean(all_best_maps):.2f} ± {np.std(all_best_maps, ddof=1):.2f}"
        )
    else:
        # 單一實驗，顯示最佳結果
        print("=" * 60)
        print(f"🏆 最佳結果: mAP = {max(all_best_maps):.4f}")
        print("=" * 60)

    # 額外：顯示所有 checkpoint 中 top 5 最高 mAP
    print("\n📈 Top 5 最高 mAP checkpoints:")
    top5 = sorted(results, key=lambda x: x["mAP"], reverse=True)[:5]
    for i, r in enumerate(top5, 1):
        print(f"  {i}. mAP={r['mAP']:.4f} | Epoch {r['epoch']:2d} | {r['exp_name']}")


if __name__ == "__main__":
    main()
