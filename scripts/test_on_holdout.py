#!/usr/bin/env python3
"""
在 Hold-out Test Set 上進行最終評估

用法:
    python scripts/test_on_holdout.py \
        --checkpoint outputs/checkpoints/siglip2_cv_run1_fold1/best_model.pth \
        --config configs/experiments/cv_experiment.yaml \
        --output_file outputs/test_results.json
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
from transformers import AutoImageProcessor, GemmaTokenizerFast
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    hamming_loss,
    coverage_error,
    label_ranking_loss,
    label_ranking_average_precision_score,
)
import numpy as np
from tqdm import tqdm


class TestSetDataset(COCOMultiLabelDataset):
    """專門用於 Test Set 的 Dataset（繼承自 COCOMultiLabelDataset）"""

    def __init__(self, data_root, processor, max_num_patches=256, text_max_length=64):
        # 不使用 K-Fold 模式，手動設置 test set
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_num_patches = max_num_patches
        self.text_max_length = text_max_length

        # 載入全域索引
        self.global_index = self._load_global_index()

        # 建立類別映射
        categories = list(self.global_index.values())[0]["categories"]
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
        self.num_classes = len(self.cat_id_to_idx)

        # 載入 test set image IDs
        self.image_ids = self._get_test_set_ids()
        print(f"✓ 測試集載入完成: {len(self.image_ids):,} 張影像, {self.num_classes} 個類別")

    def _get_test_set_ids(self) -> list:
        """獲取 Test Set 的影像 ID"""
        fold_file = self.data_root / "5fold_split.json"

        if not fold_file.exists():
            raise FileNotFoundError(
                f"5-fold split 檔案不存在: {fold_file}\n"
                "請執行: python scripts/create_kfold_split.py"
            )

        with open(fold_file) as f:
            folds_data = json.load(f)

        if "test_set" not in folds_data:
            raise ValueError("5fold_split.json 中沒有 test_set")

        return folds_data["test_set"]["image_ids"]


def create_processor(model_name: str):
    """建立 Processor（與 dataset.py 相同邏輯）"""
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = GemmaTokenizerFast.from_pretrained(model_name)

    class ProcessorWrapper:
        def __init__(self, image_processor, tokenizer):
            self.image_processor = image_processor
            self.tokenizer = tokenizer

        def __call__(self, text=None, images=None, **kwargs):
            result = {}
            return_tensors = kwargs.pop("return_tensors", "pt")

            if images is not None:
                result.update(self.image_processor(images=images, return_tensors=return_tensors))
            if text is not None:
                text_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in ["padding", "max_length", "truncation", "add_special_tokens"]
                }
                result.update(self.tokenizer(text, return_tensors=return_tensors, **text_kwargs))
            return result

    return ProcessorWrapper(image_processor, tokenizer)


@torch.no_grad()
def evaluate_on_test(model, test_loader, device="cuda"):
    """在測試集上評估，計算完整指標"""
    model.eval()

    all_logits = []
    all_labels = []

    for batch in tqdm(test_loader, desc="🔍 Evaluating on Test Set"):
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
        # mAP
        "mAP": float(average_precision_score(y_true, y_scores, average="macro")),
        "mAP_micro": float(average_precision_score(y_true, y_scores, average="micro")),
        # F1 Score
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Precision & Recall
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        # AUC-ROC (可能在某些類別缺失時失敗)
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        # Ranking 指標
        "coverage_error": float(coverage_error(y_true, y_scores)),
        "ranking_loss": float(label_ranking_loss(y_true, y_scores)),
        "lrap": float(label_ranking_average_precision_score(y_true, y_scores)),
    }

    # AUC-ROC 需要特別處理（某些類別可能沒有正樣本）
    try:
        metrics["auc_roc_macro"] = float(roc_auc_score(y_true, y_scores, average="macro"))
        metrics["auc_roc_micro"] = float(roc_auc_score(y_true, y_scores, average="micro"))
    except ValueError as e:
        print(f"⚠️ AUC-ROC 計算失敗: {e}")
        metrics["auc_roc_macro"] = None
        metrics["auc_roc_micro"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(description="在 Hold-out Test Set 上評估模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路徑")
    parser.add_argument("--config", type=str, required=True, help="配置檔案路徑")
    parser.add_argument("--output_file", type=str, default=None, help="輸出 JSON 檔案路徑")
    parser.add_argument("--device", type=str, default="cuda", help="運算裝置")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("📊 Hold-out Test Set 評估")
    print("=" * 60)

    # 載入配置
    config = OmegaConf.load(args.config)

    # 載入模型
    print(f"\n📦 載入模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = MultimodalHashKNN(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(
        f"   來自 Epoch {checkpoint.get('epoch', -1)}, "
        f"Val mAP: {checkpoint.get('val_mAP', 'N/A')}"
    )

    # 建立 Processor
    processor = create_processor(config.model.siglip2_variant)

    # 建立測試集 DataLoader
    print(f"\n📁 載入測試集...")
    test_dataset = TestSetDataset(
        data_root=config.paths.data_root,
        processor=processor,
        max_num_patches=config.model.max_num_patches,
        text_max_length=config.model.text_max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
    )

    # 評估
    print(f"\n🔍 開始評估 ({len(test_dataset):,} 張影像)...")
    metrics = evaluate_on_test(model, test_loader, args.device)

    # 輸出結果
    print("\n" + "=" * 60)
    print("📊 Hold-out Test Set 結果")
    print("=" * 60)
    print(f"  mAP (macro):     {metrics['mAP']:.4f}")
    print(f"  mAP (micro):     {metrics['mAP_micro']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (micro):      {metrics['f1_micro']:.4f}")
    print(f"  Precision:       {metrics['precision_macro']:.4f}")
    print(f"  Recall:          {metrics['recall_macro']:.4f}")
    if metrics.get("auc_roc_macro"):
        print(f"  AUC-ROC (macro): {metrics['auc_roc_macro']:.4f}")
    print(f"  Hamming Loss:    {metrics['hamming_loss']:.4f}")
    print(f"  Ranking Loss:    {metrics['ranking_loss']:.4f}")
    print(f"  LRAP:            {metrics['lrap']:.4f}")
    print("=" * 60)

    # 儲存結果
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "num_test_images": len(test_dataset),
            "metrics": metrics,
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
