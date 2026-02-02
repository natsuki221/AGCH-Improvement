#!/usr/bin/env python3
# scripts/build_knn_index.py
"""
KNN 索引建立腳本

對應手冊章節:
- §6.1 建立 Hash Index

功能:
1. 載入訓練好的模型
2. 提取訓練集所有樣本的 hash codes
3. 建立 FAISS binary index
4. 儲存索引與標籤
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 加入 src 到 Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf
from siglip2_multimodal_hash.model import MultimodalHashKNN
from siglip2_multimodal_hash.dataset import COCOMultiLabelDataset
from siglip2_multimodal_hash.knn import HashIndex
from transformers import Siglip2Processor


def parse_args():
    parser = argparse.ArgumentParser(description="建立 KNN Hash Index")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="訓練好的模型 checkpoint 路徑"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/knn_index", help="輸出索引路徑（不含副檔名）"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="推論 batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers 數量")
    parser.add_argument("--use_gpu_index", action="store_true", help="是否使用 GPU 加速 FAISS 索引")
    parser.add_argument("--fold_idx", type=int, default=None, help="K-Fold 模式下的 fold 索引")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("建立 KNN Hash Index")
    print("=" * 60)

    # 載入 checkpoint
    print(f"\n載入 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = OmegaConf.create(checkpoint["config"])

    # 建立模型
    print("\n建立模型...")
    model = MultimodalHashKNN(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✓ 模型載入完成（Epoch {checkpoint['epoch']}, Val mAP: {checkpoint['val_mAP']:.4f}）")

    # 建立 DataLoader
    print("\n建立 DataLoader...")
    processor = Siglip2Processor.from_pretrained(config.model.siglip2_variant)

    # 檢查是否使用 K-Fold
    use_k_fold = config.get("k_fold", {}).get("enabled", False) or args.fold_idx is not None

    dataset = COCOMultiLabelDataset(
        data_root=config.paths.data_root,
        processor=processor,
        max_num_patches=config.model.max_num_patches,
        text_max_length=config.model.text_max_length,
        use_k_fold=use_k_fold,
        fold_idx=args.fold_idx if use_k_fold else None,
        fold_split="train",  # 只對訓練集建立索引
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 不需要 shuffle
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"✓ 資料集: {len(dataset):,} 張影像")

    # 建立 Hash Index
    print("\n提取 Hash Codes...")
    hash_index = HashIndex(hash_bits=config.model.hash.bits, use_gpu=args.use_gpu_index)

    all_hashes = []
    all_labels = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取 Hash"):
            pixel_values = batch["pixel_values"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].numpy()
            image_ids = (
                batch["image_id"].tolist()
                if isinstance(batch["image_id"], torch.Tensor)
                else batch["image_id"]
            )

            # 提取 hash codes
            h = model.get_hash(
                pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
            )

            all_hashes.append(h.cpu().numpy())
            all_labels.append(labels)
            all_image_ids.extend(image_ids)

    # 合併
    all_hashes = np.vstack(all_hashes)
    all_labels = np.vstack(all_labels)

    print(f"\n✓ 提取完成:")
    print(f"  - Hash codes: {all_hashes.shape}")
    print(f"  - Labels: {all_labels.shape}")
    print(f"  - Image IDs: {len(all_image_ids)}")

    # 添加到索引
    print("\n建立索引...")
    hash_index.add(all_hashes, all_labels, all_image_ids)

    # 儲存
    output_path = Path(args.output)
    if args.fold_idx is not None:
        output_path = output_path.with_name(f"{output_path.name}_fold{args.fold_idx}")

    hash_index.save(output_path)

    print("\n" + "=" * 60)
    print("✓ KNN Hash Index 建立完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
