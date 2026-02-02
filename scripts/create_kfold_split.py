"""
生成 5-Fold Cross-Validation 的靜態切分索引

輸出: data/coco/5fold_split.json
結構:
{
    "fold_0": {"train": [id1, id2, ...], "val": [id3, id4, ...]},
    "fold_1": {...},
    ...
}
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

# 設定路徑
DATA_ROOT = Path("./data/coco")
KARPATHY_PATH = DATA_ROOT / "karpathy_split.json"
OUTPUT_PATH = DATA_ROOT / "5fold_split.json"


def create_folds():
    """生成 5-fold 切分"""

    print(f"正在讀取: {KARPATHY_PATH}")

    if not KARPATHY_PATH.exists():
        raise FileNotFoundError(
            f"Karpathy split 檔案不存在: {KARPATHY_PATH}\n"
            "請執行: python scripts/download_karpathy_split.py"
        )

    with open(KARPATHY_PATH) as f:
        data = json.load(f)

    # 1. 建立開發集池（排除 test）
    dev_pool = []
    test_pool = []
    skipped_restval = 0

    for img in data["images"]:
        # 解析 COCO ID
        filename = img["filename"]
        img_id = int(filename.split("_")[-1].split(".")[0])

        split_type = img.get("split", "unknown")

        if split_type == "test":
            test_pool.append(img_id)
        elif split_type in ["train", "val"]:
            dev_pool.append(img_id)
        elif split_type == "restval":
            # restval 通常已包含在 train 中，這裡跳過以避免重複
            skipped_restval += 1
        else:
            print(f"⚠️  未知的 split 類型: {split_type} (Image ID: {img_id})")

    print(f"開發集池總數: {len(dev_pool)}")
    print(f"測試集總數: {len(test_pool)} (已排除)")
    print(f"跳過 restval: {skipped_restval} (避免重複)")

    # 2. 5-Fold 切分
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    dev_pool_np = np.array(dev_pool)

    folds_data = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dev_pool_np)):
        fold_name = f"fold_{fold_idx}"
        folds_data[fold_name] = {
            "train": dev_pool_np[train_idx].tolist(),
            "val": dev_pool_np[val_idx].tolist(),
        }
        print(f" - {fold_name}: Train={len(train_idx):,}, Val={len(val_idx):,}")

    # 3. 儲存測試集資訊（供最終評估使用）
    folds_data["test_set"] = {
        "image_ids": test_pool,
        "note": "Hold-out test set, DO NOT USE during training!",
    }

    # 4. 存檔
    with open(OUTPUT_PATH, "w") as f:
        json.dump(folds_data, f, indent=2)

    print(f"\n✓ 已儲存至: {OUTPUT_PATH}")
    print(f"  - 開發集: {len(dev_pool):,} 張（用於 5-fold CV）")
    print(f"  - 測試集: {len(test_pool):,} 張（最終評估）")

    # 5. 驗證
    total_cv_samples = sum(
        len(folds_data[f"fold_{i}"]["train"]) + len(folds_data[f"fold_{i}"]["val"])
        for i in range(5)
    )
    assert total_cv_samples == len(dev_pool) * 5, "Fold 數量驗證失敗"
    print(f"✓ 驗證通過：每個樣本在 5 折中出現了 1 次訓練 + 1 次驗證")


if __name__ == "__main__":
    create_folds()
