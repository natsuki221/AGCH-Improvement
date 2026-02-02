#!/usr/bin/env python3
"""
在 Hold-out Test Set 上進行最終評估

用法:
    python scripts/test_on_holdout.py \
        --checkpoint outputs/checkpoints/siglip2_cv_run1_fold0/best_model.pth \
        --config configs/experiments/cv_experiment.yaml
"""

import torch
import argparse
from pathlib import Path
import sys
import json
from omegaconf import OmegaConf

# 加入 src 到 Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from siglip2_multimodal_hash.model import MultimodalHashKNN
from siglip2_multimodal_hash.dataset import COCOMultiLabelDataset
from torch.utils.data import DataLoader
from transformers import Siglip2Processor
from sklearn.metrics import average_precision_score, f1_score
import numpy as np
from tqdm import tqdm


def load_test_set(config):
    """載入測試集"""
    # 讀取 5fold_split.json 以獲取測試集 ID
    fold_file = Path(config.paths.data_root) / "5fold_split.json"

    with open(fold_file) as f:
        folds_data = json.load(f)

    test_ids = folds_data["test_set"]["image_ids"]

    print(f"載入測試集: {len(test_ids)} 張影像")

    # 建立自定義 Dataset（僅包含測試集 ID）
    # TODO: 需要修改 COCOMultiLabelDataset 支援自定義 image_ids
    # 這裡簡化實作

    return test_ids


@torch.no_grad()
def evaluate_on_test(model, test_loader, device="cuda"):
    """在測試集上評估"""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        # 前向傳播
        logits = model(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

        all_logits.append(logits.cpu())
        all_labels.append(labels)

    # 合併結果
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 計算指標
    y_true = all_labels.numpy()
    y_scores = torch.sigmoid(all_logits).numpy()
    y_pred = (y_scores > 0.5).astype(int)

    metrics = {
        "mAP": average_precision_score(y_true, y_scores, average="macro"),
        "mAP_micro": average_precision_score(y_true, y_scores, average="micro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路徑")
    parser.add_argument("--config", type=str, required=True, help="配置檔案路徑")
    parser.add_argument("--device", type=str, default="cuda", help="運算裝置")
    args = parser.parse_args()

    # 載入配置
    config = OmegaConf.load(args.config)

    # 載入模型
    print(f"載入模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = MultimodalHashKNN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(
        f"模型來自 Epoch {checkpoint.get('epoch', -1)}, "
        f"Val mAP: {checkpoint.get('val_mAP', 'N/A')}"
    )

    # 載入測試集
    # TODO: 實作測試集 DataLoader
    # test_loader = ...

    # 評估
    # metrics = evaluate_on_test(model, test_loader, args.device)

    # print("\n" + "="*60)
    # print("📊 Hold-out Test Set 結果")
    # print("="*60)
    # print(f"mAP (macro): {metrics['mAP']:.4f}")
    # print(f"mAP (micro): {metrics['mAP_micro']:.4f}")
    # print(f"F1 (macro):  {metrics['f1_macro']:.4f}")
    # print(f"F1 (micro):  {metrics['f1_micro']:.4f}")
    # print("="*60)

    print("⚠️  此腳本為範例，需要完整實作測試集載入邏輯")


if __name__ == "__main__":
    main()
