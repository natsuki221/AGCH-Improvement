"""下載並處理 Karpathy split for COCO"""

import json
import urllib.request
from pathlib import Path

# Karpathy split URL
KARPATHY_URL = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

# 或使用 GitHub 備份
GITHUB_URL = "https://raw.githubusercontent.com/karpathy/neuraltalk2/master/coco/cocotalk.json"

data_dir = Path("./data/coco")
data_dir.mkdir(parents=True, exist_ok=True)

print("正在下載 Karpathy split...")

# 下載 JSON 檔案
output_file = data_dir / "karpathy_split.json"

try:
    urllib.request.urlretrieve(GITHUB_URL, output_file)
    print(f"✓ 下載成功: {output_file}")
except Exception as e:
    print(f"✗ 下載失敗: {e}")
    print("請手動下載: https://www.kaggle.com/datasets/shtvkumar/karpathy-splits")
    exit(1)

# 載入並驗證
with open(output_file) as f:
    data = json.load(f)

# 統計分割
splits = {"train": 0, "val": 0, "test": 0, "restval": 0}
for item in data["images"]:
    split = item.get("split", "unknown")
    splits[split] = splits.get(split, 0) + 1

print("\nKarpathy Split 統計:")
for split, count in splits.items():
    print(f"  {split}: {count}")

# 驗證
assert splits["train"] == 113287, f"訓練集數量錯誤: {splits['train']} (預期 113287)"
assert splits["val"] == 5000, f"驗證集數量錯誤: {splits['val']} (預期 5000)"
assert splits["test"] == 5000, f"測試集數量錯誤: {splits['test']} (預期 5000)"

print("\n✓ Karpathy split 驗證通過！")