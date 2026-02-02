# src/siglip2_multimodal_hash/dataset.py

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pickle
import random
import json
from typing import Optional


class COCOMultiLabelDataset(Dataset):
    """COCO 多標籤資料集（支援 5-Fold CV）"""

    def __init__(
        self,
        data_root: str | Path,
        processor=None,
        max_num_patches: int = 256,
        text_max_length: int = 64,
        # 5-Fold CV 參數
        use_k_fold: bool = False,
        fold_idx: Optional[int] = None,
        fold_split: str = "train",  # "train" or "val"
    ):
        """
        Args:
            data_root: COCO 資料集根目錄
            processor: SigLIP2Processor
            max_num_patches: 最大 patch 數量
            text_max_length: 文字最大長度
            use_k_fold: 是否使用 K-Fold 模式
            fold_idx: Fold 索引（0-4）
            fold_split: 當前 Fold 的角色（train/val）
        """
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_num_patches = max_num_patches
        self.text_max_length = text_max_length

        # 載入兩個索引（因為 Dev Pool 橫跨 train2014 和 val2014）
        self.global_index = self._load_global_index()

        # 建立類別映射
        categories = list(self.global_index.values())[0]["categories"]
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
        self.num_classes = len(self.cat_id_to_idx)

        # 決定影像 ID 列表
        if use_k_fold:
            self.image_ids = self._get_fold_image_ids(fold_idx, fold_split)
            print(
                f"✓ K-Fold 模式: Fold {fold_idx}, Split={fold_split}, "
                f"影像數={len(self.image_ids):,}"
            )
        else:
            # 傳統模式（使用 Karpathy split）
            self.image_ids = self._get_karpathy_image_ids()
            print(f"✓ 傳統模式: 影像數={len(self.image_ids):,}")

        print(f"✓ {self.num_classes} 個類別")

    def _load_global_index(self) -> dict:
        """載入全域索引（合併 train + val）"""
        train_index_file = self.data_root / "index_train2014.pkl"
        val_index_file = self.data_root / "index_val2014.pkl"

        with open(train_index_file, "rb") as f:
            train_index = pickle.load(f)

        with open(val_index_file, "rb") as f:
            val_index = pickle.load(f)

        # 合併
        global_index = {}

        # Train images
        for img_id, img_info in train_index["images"].items():
            img_info["physical_split"] = "train2014"  # 記錄物理位置
            global_index[img_id] = {"info": img_info, "categories": train_index["categories"]}

        # Val images
        for img_id, img_info in val_index["images"].items():
            img_info["physical_split"] = "val2014"
            global_index[img_id] = {"info": img_info, "categories": val_index["categories"]}

        print(f"載入全域索引: {len(global_index):,} 張影像")
        return global_index

    def _get_fold_image_ids(self, fold_idx: int, fold_split: str) -> list:
        """獲取指定 Fold 的影像 ID"""
        fold_file = self.data_root / "5fold_split.json"

        if not fold_file.exists():
            raise FileNotFoundError(
                f"5-fold split 檔案不存在: {fold_file}\n"
                "請執行: python scripts/create_kfold_split.py"
            )

        with open(fold_file) as f:
            folds_data = json.load(f)

        fold_name = f"fold_{fold_idx}"
        if fold_name not in folds_data:
            raise ValueError(f"Fold {fold_idx} 不存在（有效範圍: 0-4）")

        image_ids = folds_data[fold_name][fold_split]
        return image_ids

    def _get_karpathy_image_ids(self) -> list:
        """傳統 Karpathy split（用於對比實驗）"""
        karpathy_file = self.data_root / "karpathy_split.json"

        with open(karpathy_file) as f:
            data = json.load(f)

        # 預設使用 train split
        image_ids = []
        for item in data["images"]:
            if item.get("split") == "train":
                filename = item["filename"]
                img_id = int(filename.split("_")[-1].split(".")[0])
                image_ids.append(img_id)

        return image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]

        # 從全域索引中獲取資訊
        img_data = self.global_index[img_id]
        img_info = img_data["info"]

        # 載入影像（根據物理位置）
        physical_split = img_info["physical_split"]
        img_path = self.data_root / "images" / physical_split / img_info["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"影像不存在: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 隨機選擇一個 caption
        caption = random.choice(img_info["captions"])

        # 建立 multi-hot label
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for cat_id in img_info["categories"]:
            labels[self.cat_id_to_idx[cat_id]] = 1.0

        # 使用 processor 處理
        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "image_id": img_id,
            "caption": caption,
        }


def create_dataloader(config, split: str = "train", fold_idx: Optional[int] = None):
    """建立 DataLoader（支援 K-Fold）"""
    from torch.utils.data import DataLoader
    from transformers import Siglip2Processor

    # 載入 processor
    processor = Siglip2Processor.from_pretrained(config.model.siglip2_variant)

    # 檢查是否使用 K-Fold
    use_k_fold = config.get("k_fold", {}).get("enabled", False)

    if use_k_fold and fold_idx is None:
        raise ValueError("K-Fold 模式下必須指定 fold_idx")

    # 建立 dataset
    dataset = COCOMultiLabelDataset(
        data_root=config.paths.data_root,
        processor=processor,
        max_num_patches=config.model.max_num_patches,
        text_max_length=config.model.text_max_length,
        use_k_fold=use_k_fold,
        fold_idx=fold_idx,
        fold_split=split,  # "train" or "val"
    )

    # 建立 dataloader
    shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        prefetch_factor=config.dataloader.prefetch_factor,
        persistent_workers=config.dataloader.persistent_workers,
        drop_last=config.dataloader.drop_last if split == "train" else False,
    )

    return dataloader
