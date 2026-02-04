#!/usr/bin/env python3
# scripts/evaluate.py
"""
模型評估腳本

對應手冊章節:
- §9.3 驗證迴圈
- §12 評估指標

功能:
1. 載入模型與 KNN 索引
2. 在驗證/測試集上進行推論
3. 計算 mAP、Precision、Recall、F1 等指標
4. 輸出評估報告
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import json

# 加入 src 到 Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf
from siglip2_multimodal_hash.model import MultimodalHashKNN
from siglip2_multimodal_hash.dataset import COCOMultiLabelDataset
from siglip2_multimodal_hash.knn import HashIndex, predict_tags
from siglip2_multimodal_hash.losses import compute_total_loss
from transformers import Siglip2Processor


def parse_args():
    parser = argparse.ArgumentParser(description="模型評估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路徑")
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="KNN 索引路徑（不含副檔名）。如果提供，使用 KNN 推論",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="評估的資料集切分",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="推論 batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers 數量")
    parser.add_argument("--threshold", type=float, default=0.5, help="分類閾值")
    parser.add_argument("--k", type=int, default=20, help="KNN 鄰居數量")
    parser.add_argument("--tau", type=float, default=0.07, help="KNN softmax 溫度參數")
    parser.add_argument("--fold_idx", type=int, default=None, help="K-Fold 模式下的 fold 索引")
    parser.add_argument("--output", type=str, default=None, help="評估結果輸出路徑（JSON）")
    parser.add_argument("--use_knn", action="store_true", help="使用 KNN 推論（需要 --index）")
    return parser.parse_args()


def evaluate_with_classifier(
    model: MultimodalHashKNN, dataloader: DataLoader, loss_config, threshold: float = 0.5
) -> dict:
    """
    使用分類器頭進行評估

    Returns:
        評估指標字典
    """
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0

    print("\n使用分類器頭進行評估...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="評估中"):
            pixel_values = batch["pixel_values"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            with autocast(dtype=torch.float16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_components=True,
                )

                loss_dict = compute_total_loss(outputs, labels, loss_config)

            total_loss += loss_dict["total"].item()
            all_logits.append(outputs["logits"].cpu())
            all_labels.append(labels.cpu())

    # 合併
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 計算指標
    y_true = all_labels.numpy()
    y_scores = torch.sigmoid(all_logits).numpy()
    y_pred = (y_scores > threshold).astype(int)

    metrics = {
        "loss": total_loss / len(dataloader),
        "mAP_macro": average_precision_score(y_true, y_scores, average="macro"),
        "mAP_micro": average_precision_score(y_true, y_scores, average="micro"),
        "mAP_weighted": average_precision_score(y_true, y_scores, average="weighted"),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics


def evaluate_with_knn(
    model: MultimodalHashKNN,
    dataloader: DataLoader,
    hash_index: HashIndex,
    k: int = 20,
    tau: float = 0.07,
    threshold: float = 0.5,
) -> dict:
    """
    使用 KNN 進行評估

    Returns:
        評估指標字典
    """
    model.eval()

    all_hashes = []
    all_labels = []

    print("\n使用 KNN 進行評估...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取 Hash"):
            pixel_values = batch["pixel_values"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].numpy()

            h = model.get_hash(
                pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
            )

            all_hashes.append(h.cpu().numpy())
            all_labels.append(labels)

    # 合併
    all_hashes = np.vstack(all_hashes)
    all_labels = np.vstack(all_labels)

    print(f"\n進行 KNN 推論 (K={k}, tau={tau})...")

    # KNN 推論
    predictions = predict_tags(
        query_hash=all_hashes,
        index=hash_index,
        k=k,
        tau=tau,
        voting_strategy="softmax",
        top_n=10,  # 返回 Top-10
    )

    # 計算指標
    y_true = all_labels
    y_scores = predictions["tag_scores"]
    y_pred = (y_scores > threshold).astype(int)

    metrics = {
        "mAP_macro": average_precision_score(y_true, y_scores, average="macro"),
        "mAP_micro": average_precision_score(y_true, y_scores, average="micro"),
        "mAP_weighted": average_precision_score(y_true, y_scores, average="weighted"),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "knn_k": k,
        "knn_tau": tau,
    }

    return metrics


def print_metrics(metrics: dict, title: str = "評估結果"):
    """格式化輸出評估指標"""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)

    if "loss" in metrics:
        print(f"\n損失: {metrics['loss']:.4f}")

    print("\n── mAP ──")
    print(f"  Macro:    {metrics['mAP_macro']:.4f}")
    print(f"  Micro:    {metrics['mAP_micro']:.4f}")
    print(f"  Weighted: {metrics['mAP_weighted']:.4f}")

    print("\n── F1 Score ──")
    print(f"  Macro:    {metrics['f1_macro']:.4f}")
    print(f"  Micro:    {metrics['f1_micro']:.4f}")
    print(f"  Weighted: {metrics['f1_weighted']:.4f}")

    print("\n── Precision ──")
    print(f"  Macro:    {metrics['precision_macro']:.4f}")
    print(f"  Micro:    {metrics['precision_micro']:.4f}")

    print("\n── Recall ──")
    print(f"  Macro:    {metrics['recall_macro']:.4f}")
    print(f"  Micro:    {metrics['recall_micro']:.4f}")

    if "knn_k" in metrics:
        print(f"\n── KNN 參數 ──")
        print(f"  K:   {metrics['knn_k']}")
        print(f"  τ:   {metrics['knn_tau']}")

    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("模型評估")
    print("=" * 60)

    # 載入 checkpoint
    print(f"\n載入 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = OmegaConf.create(checkpoint["config"])

    # 建立模型
    print("\n建立模型...")
    model = MultimodalHashKNN(config.model).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"✓ 模型載入完成（Epoch {checkpoint.get('epoch', 'N/A')}, "
        f"Val mAP: {checkpoint.get('val_mAP', 'N/A'):.4f}）"
    )

    # 建立 DataLoader
    print(f"\n建立 DataLoader (split={args.split})...")

    # 分開載入 processor 組件（避開 Siglip2Processor tokenizer bug）
    from transformers import AutoImageProcessor, GemmaTokenizerFast

    model_name = config.model.siglip2_variant
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

    processor = ProcessorWrapper(image_processor, tokenizer)

    use_k_fold = config.get("k_fold", {}).get("enabled", False) or args.fold_idx is not None

    dataset = COCOMultiLabelDataset(
        data_root=config.paths.data_root,
        processor=processor,
        max_num_patches=config.model.max_num_patches,
        text_max_length=config.model.text_max_length,
        use_k_fold=use_k_fold,
        fold_idx=args.fold_idx if use_k_fold else None,
        fold_split=args.split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"✓ 資料集: {len(dataset):,} 張影像")

    # 選擇評估方式
    if args.use_knn and args.index:
        # KNN 評估
        print(f"\n載入 KNN 索引: {args.index}")
        hash_index = HashIndex.load(args.index)

        metrics = evaluate_with_knn(
            model=model,
            dataloader=dataloader,
            hash_index=hash_index,
            k=args.k,
            tau=args.tau,
            threshold=args.threshold,
        )
        title = f"KNN 評估結果 (K={args.k}, τ={args.tau})"
    else:
        # 分類器評估
        metrics = evaluate_with_classifier(
            model=model, dataloader=dataloader, loss_config=config.loss, threshold=args.threshold
        )
        title = "分類器評估結果"

    # 輸出結果
    print_metrics(metrics, title)

    # 儲存結果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ 結果已儲存: {output_path}")

    print("\n評估完成！")


if __name__ == "__main__":
    main()
