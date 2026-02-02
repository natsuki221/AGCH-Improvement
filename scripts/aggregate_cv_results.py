#!/usr/bin/env python3
"""
聚合 5-Fold Cross-Validation 的實驗結果

用法:
    python scripts/aggregate_cv_results.py --exp_prefix siglip2_cv_run1_fold
"""

import argparse
import glob
import torch
import numpy as np
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser(description="聚合 5-Fold CV 結果")
    parser.add_argument("--exp_prefix", type=str, required=True,
                        help="實驗名稱前綴（例如: siglip2_cv_run1_fold）")
    parser.add_argument("--results_dir", type=str, default="outputs/checkpoints",
                        help="結果目錄")
    parser.add_argument("--output_file", type=str, default="cv_results_summary.json",
                        help="輸出檔案名稱")
    args = parser.parse_args()
    
    # 搜尋所有 fold 的模型
    pattern = f"{args.results_dir}/{args.exp_prefix}*/*.pth"
    files = glob.glob(pattern)
    
    scores = {}
    fold_results = []
    
    print(f"搜尋路徑: {pattern}")
    print(f"找到 {len(files)} 個模型檔案:\n")
    
    for f in sorted(files):
        try:
            # 載入 checkpoint
            ckpt = torch.load(f, map_location="cpu")
            
            # 提取資訊
            fold_name = Path(f).parent.name
            val_mAP = ckpt.get('val_mAP', None)
            epoch = ckpt.get('epoch', -1)
            
            if val_mAP is None:
                print(f" ⚠️  {fold_name}: 無 val_mAP 資訊，跳過")
                continue
            
            # 提取 fold 編號
            import re
            match = re.search(r'fold(\d+)', fold_name)
            if match:
                fold_idx = int(match.group(1))
            else:
                fold_idx = -1
            
            fold_info = {
                "fold": fold_idx,
                "fold_name": fold_name,
                "epoch": epoch,
                "val_mAP": val_mAP,
                "checkpoint_path": str(f)
            }
            
            fold_results.append(fold_info)
            scores[fold_idx] = val_mAP
            
            print(f" - Fold {fold_idx}: Epoch {epoch}, mAP = {val_mAP:.4f}")
            
        except Exception as e:
            print(f" ❌ 讀取錯誤 {f}: {e}")
    
    # 計算統計量
    if scores:
        values = list(scores.values())
        
        print("\n" + "="*60)
        print(f"🏆 5-Fold Cross-Validation 最終結果 (mAP)")
        print("="*60)
        print(f"Mean: {np.mean(values):.4f}")
        print(f"Std:  {np.std(values, ddof=1):.4f}")  # 使用樣本標準差
        print(f"Min:  {np.min(values):.4f}")
        print(f"Max:  {np.max(values):.4f}")
        print(f"Median: {np.median(values):.4f}")
        print("="*60)
        
        # 論文格式
        mean_map = np.mean(values)
        std_map = np.std(values, ddof=1)
        print(f"\n📝 論文報告格式:")
        print(f"   mAP: {mean_map:.2f} ± {std_map:.2f}")
        print("="*60)
        
        # 儲存結果
        summary = {
            "experiment": args.exp_prefix,
            "num_folds": len(scores),
            "statistics": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            },
            "fold_results": fold_results
        }
        
        output_path = Path(args.results_dir) / args.output_file
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ 結果已儲存至: {output_path}")
        
    else:
        print("\n⚠️  未找到有效結果。")

if __name__ == "__main__":
    main()