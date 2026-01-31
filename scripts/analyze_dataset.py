import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

def analyze(data_dir, split="train2014"):
    """分析資料集統計"""
    
    # 載入索引
    index_file = Path(data_dir) / f"index_{split}.pkl"
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"{split.upper()} 資料集統計")
    print(f"{'='*60}")
    
    # 1. 基本統計
    n_images = len(index["images"])
    n_categories = len(index["categories"])
    
    print(f"\n📊 基本統計:")
    print(f"  影像數量: {n_images:,}")
    print(f"  類別數量: {n_categories}")
    
    # 2. 每張圖的標籤數量分布
    labels_per_image = [len(info["categories"]) for info in index["images"].values()]
    
    print(f"\n🏷️  標籤數量分布:")
    print(f"  平均每張圖的標籤數: {np.mean(labels_per_image):.2f}")
    print(f"  中位數: {np.median(labels_per_image):.0f}")
    print(f"  最小值: {np.min(labels_per_image)}")
    print(f"  最大值: {np.max(labels_per_image)}")
    
    # 3. 類別頻率
    category_counts = Counter()
    for img_info in index["images"].values():
        category_counts.update(img_info["categories"])
    
    print(f"\n🔝 Top 10 最常出現的類別:")
    for cat_id, count in category_counts.most_common(10):
        cat_name = index["categories"][cat_id]["name"]
        percentage = count / n_images * 100
        print(f"  {cat_name:20s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n🔻 Bottom 10 最少出現的類別:")
    for cat_id, count in category_counts.most_common()[-10:]:
        cat_name = index["categories"][cat_id]["name"]
        percentage = count / n_images * 100
        print(f"  {cat_name:20s}: {count:6,} ({percentage:5.2f}%)")
    
    # 4. Caption 長度分布
    caption_lengths = []
    for img_info in index["images"].values():
        for cap in img_info["captions"]:
            caption_lengths.append(len(cap.split()))
    
    print(f"\n📝 Caption 統計:")
    print(f"  平均長度: {np.mean(caption_lengths):.2f} 個字")
    print(f"  中位數: {np.median(caption_lengths):.0f} 個字")
    print(f"  最短: {np.min(caption_lengths)} 個字")
    print(f"  最長: {np.max(caption_lengths)} 個字")
    
    return {
        "labels_per_image": labels_per_image,
        "category_counts": category_counts,
        "caption_lengths": caption_lengths,
    }

if __name__ == "__main__":
    data_dir = Path("./data/coco")
    
    # 分析訓練集
    train_stats = analyze(data_dir, "train2014")
    
    # 分析驗證集
    val_stats = analyze(data_dir, "val2014")
    
    # 視覺化 (可選)
    # ... (可自行加入 matplotlib 繪圖)