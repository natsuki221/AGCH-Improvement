#!/usr/bin/env python3
"""
聚合 5-Fold Cross-Validation 的實驗結果
支援所有多標籤分類研究常用指標

用法:
    python scripts/aggregate_cv_results.py --exp_prefix siglip2_cv_run1_fold
"""

import argparse
import glob
import torch
import numpy as np
from pathlib import Path
import json
import re

# 所有追蹤的指標及其方向（True=越高越好，False=越低越好）
METRICS = {
    "mAP": ("mAP (↑)", True),
    "auc_macro": ("AUC-Macro (↑)", True),
    "auc_micro": ("AUC-Micro (↑)", True),
    "f1_macro": ("F1-Macro (↑)", True),
    "f1_micro": ("F1-Micro (↑)", True),
    "precision_macro": ("Precision-Macro (↑)", True),
    "precision_micro": ("Precision-Micro (↑)", True),
    "recall_macro": ("Recall-Macro (↑)", True),
    "recall_micro": ("Recall-Micro (↑)", True),
    "lrap": ("LRAP (↑)", True),
    "hamming_loss": ("Hamming Loss (↓)", False),
    "ranking_loss": ("Ranking Loss (↓)", False),
    "coverage_error": ("Coverage Error (↓)", False),
    "mae": ("MAE (↓)", False),
}


def compute_stats(values):
    """計算統計量"""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="聚合 5-Fold CV 結果")
    parser.add_argument(
        "--exp_prefix", type=str, required=True, help="實驗名稱前綴（例如: siglip2_cv_run1_fold）"
    )
    parser.add_argument("--results_dir", type=str, default="outputs/checkpoints", help="結果目錄")
    parser.add_argument(
        "--output_file", type=str, default="cv_results_summary.json", help="輸出檔案名稱"
    )
    args = parser.parse_args()

    # 搜尋所有 fold 的模型
    pattern = f"{args.results_dir}/{args.exp_prefix}*/*.pth"
    files = glob.glob(pattern)

    print(f"搜尋路徑: {pattern}")
    print(f"找到 {len(files)} 個模型檔案\n")

    if not files:
        print("⚠️ 未找到有效結果。")
        return

    # 收集每個 fold 的所有指標
    fold_results = []
    metrics_by_fold = {m: {} for m in METRICS.keys()}

    for f in sorted(files):
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)

            fold_name = Path(f).parent.name
            epoch = ckpt.get("epoch", -1)

            # 嘗試從 val_metrics 取得完整指標（新格式）
            val_metrics = ckpt.get("val_metrics", None)

            # 向後兼容：從單獨欄位取得
            if val_metrics is None:
                val_metrics = {"mAP": ckpt.get("val_mAP", None)}

            if val_metrics.get("mAP") is None:
                print(f" ⚠️ {fold_name}: 無 mAP 資訊，跳過")
                continue

            # 提取 fold 編號
            match = re.search(r"fold(\d+)", fold_name)
            fold_idx = int(match.group(1)) if match else -1

            # 收集結果
            fold_info = {
                "fold": fold_idx,
                "fold_name": fold_name,
                "epoch": epoch,
                "checkpoint_path": str(f),
                "metrics": {},
            }

            for metric_key in METRICS.keys():
                value = val_metrics.get(metric_key)
                if value is not None:
                    fold_info["metrics"][metric_key] = float(value)
                    metrics_by_fold[metric_key][fold_idx] = float(value)

            fold_results.append(fold_info)

            # 顯示每個 fold 的結果
            mAP = val_metrics.get("mAP", 0)
            auc = val_metrics.get("auc_macro", 0)
            f1 = val_metrics.get("f1_macro", 0)
            print(
                f" - Fold {fold_idx}: Epoch {epoch:2d} | mAP={mAP:.4f} | AUC={auc:.4f} | F1={f1:.4f}"
            )

        except Exception as e:
            print(f" ❌ 讀取錯誤 {f}: {e}")

    if not fold_results:
        print("\n⚠️ 未找到有效結果。")
        return

    # 計算所有指標的統計量
    all_stats = {}

    print("\n" + "=" * 70)
    print("🏆 5-Fold Cross-Validation 最終結果")
    print("=" * 70)

    # 主要指標表格
    print("\n📊 主要指標:")
    print("-" * 50)
    print(f"{'指標':<25} {'Mean':>10} {'± Std':>10}")
    print("-" * 50)

    for metric_key, (display_name, higher_better) in METRICS.items():
        values = list(metrics_by_fold[metric_key].values())
        if values:
            stats = compute_stats(values)
            all_stats[metric_key] = stats
            print(f"{display_name:<25} {stats['mean']:>10.4f} {stats['std']:>9.4f}")

    print("-" * 50)

    # 論文格式
    print("\n📝 論文報告格式:")
    print("=" * 70)

    key_metrics = ["mAP", "auc_macro", "f1_macro", "precision_macro", "recall_macro"]
    for m in key_metrics:
        if m in all_stats:
            stats = all_stats[m]
            display = METRICS[m][0].split(" ")[0]
            print(f"   {display}: {stats['mean']:.2f} ± {stats['std']:.2f}")

    print("=" * 70)

    # 儲存完整結果
    summary = {
        "experiment": args.exp_prefix,
        "num_folds": len(fold_results),
        "statistics": all_stats,
        "fold_results": fold_results,
    }

    output_path = Path(args.results_dir) / args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 完整結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
