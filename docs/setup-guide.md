# SigLIP2 多模態分類實驗環境設置完整教學

> **使用工具**: `uv` - 超快速的 Python 套件管理器  
> **目標**: 建立可重現、隔離、高效的實驗環境

---

## 目錄

1. [系統需求檢查](#1-系統需求檢查)
2. [安裝 uv](#2-安裝-uv)
3. [專案初始化](#3-專案初始化)
4. [環境配置與依賴安裝](#4-環境配置與依賴安裝)
5. [資料集下載與處理](#5-資料集下載與處理)
6. [驗證環境](#6-驗證環境)
7. [常見問題排除](#7-常見問題排除)

---

## 1) 系統需求檢查

### 1.1 硬體需求

在開始之前，請確認你的系統滿足以下需求：

```bash
# 檢查 GPU
nvidia-smi

# 應該看到類似輸出：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2   |
# | -------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# | 30%   45C    P8    25W / 320W |    500MiB / 24576MiB |      0%      Default |
```

**最低需求**:

- ✅ GPU: NVIDIA RTX 3090 或更高 (24GB VRAM)
- ✅ CPU: 8 核心或更多
- ✅ RAM: 32 GB 或更多
- ✅ Storage: 至少 100 GB 可用空間 (建議 SSD)

### 1.2 軟體需求

```bash
# 檢查 CUDA 版本
nvcc --version

# 應該顯示 CUDA 11.8 或更高
# 如果沒有 nvcc，檢查：
cat /usr/local/cuda/version.txt

# 檢查 Python 版本
python3 --version
# 需要 Python 3.10 或更高（建議 3.11）
```

**如果 Python 版本不符合**:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# 或使用 pyenv (推薦)
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7
```

---

## 2) 安裝 uv

### 2.1 什麼是 uv？

`uv` 是由 Astral (Ruff 的開發者) 推出的超快速 Python 套件管理器：

- ⚡ 比 pip 快 10-100 倍
- 🔒 內建依賴解析與鎖定
- 📦 自動管理虛擬環境
- 🎯 完全相容 pip/requirements.txt

### 2.2 安裝 uv

```bash
# 方法 1: 使用官方安裝腳本 (推薦)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法 2: 使用 pip (如果已有 Python)
pip install uv

# 方法 3: 使用 cargo (如果你是 Rust 開發者)
cargo install --git https://github.com/astral-sh/uv uv
```

### 2.3 驗證安裝

```bash
# 檢查版本
uv --version
# 應該顯示: uv 0.1.x 或更高

# 查看幫助
uv --help
```

### 2.4 配置 uv (可選但推薦)

```bash
# 設定鏡像源加速下載 (中國用戶)
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 或在配置文件中永久設定
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << EOF
[pip]
index-url = "https://pypi.org/simple"
extra-index-url = [
    "https://download.pytorch.org/whl/cu121"
]
EOF
```

---

## 3) 專案初始化

### 3.1 建立專案目錄結構

```bash
# 建立專案根目錄
mkdir -p ~/projects/siglip2-multimodal-hash
cd ~/projects/siglip2-multimodal-hash

# 建立標準目錄結構
mkdir -p {src,data,experiments,outputs,notebooks,scripts,configs,tests}

# 專案結構說明：
# src/           - 主要原始碼
# data/          - 資料集存放位置
# experiments/   - 實驗結果與日誌
# outputs/       - 模型輸出與檢查點
# notebooks/     - Jupyter notebooks (探索性分析)
# scripts/       - 獨立腳本 (下載、預處理等)
# configs/       - 配置檔案 (YAML)
# tests/         - 單元測試
```

### 3.2 初始化 Git (強烈推薦)

```bash
# 初始化 git
git init

# 建立 .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data (不要上傳大型資料集)
data/raw/
data/coco/
*.zip
*.tar.gz

# Experiments
experiments/*/
outputs/checkpoints/
*.pth
*.pt
*.ckpt

# Logs
*.log
logs/
wandb/

# OS
.DS_Store
Thumbs.db
EOF

# 建立 README
cat > README.md << 'EOF'
# SigLIP2 Multimodal Hash-based Multi-label Classification

基於 SigLIP2 的多模態圖文多標籤分類系統，結合深度 hashing 與 KNN 檢索。

## 專案結構
見 `docs/setup_guide.md`

## 快速開始
見 `docs/setup_guide.md`
EOF

# 首次提交
git add .gitignore README.md
git commit -m "Initial commit: project structure"
```

---

## 4) 環境配置與依賴安裝

### 4.1 建立 pyproject.toml

```bash
# 建立現代化的 Python 專案配置
cat > pyproject.toml << 'EOF'
[project]
name = "siglip2-multimodal-hash"
version = "0.1.0"
description = "Multimodal image-text multi-label classification using SigLIP2, hashing, and KNN"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

dependencies = [
    # Deep Learning Framework
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    
    # Transformers & Vision-Language Models
    "transformers>=4.40.0",
    "accelerate>=0.27.0",
    
    # Computer Vision
    "opencv-python>=4.9.0",
    "pillow>=10.2.0",
    "albumentations>=1.4.0",
    
    # Data Processing
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "pycocotools>=2.0.7",
    
    # Similarity Search & Indexing
    "faiss-gpu>=1.7.2",  # GPU 版本
    # "faiss-cpu>=1.7.2",  # 如果沒有 GPU 則用此行
    
    # Metrics & Evaluation
    "scikit-learn>=1.4.0",
    "scipy>=1.12.0",
    
    # Visualization
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    
    # Configuration & Logging
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
    
    # Utilities
    "tqdm>=4.66.0",
    "rich>=13.7.0",
    "python-dotenv>=1.0.0",
    
    # Development Tools
    "ipython>=8.20.0",
    "jupyter>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "black>=24.0.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py310', 'py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
EOF
```

### 4.2 使用 uv 建立虛擬環境並安裝依賴

```bash
# 建立虛擬環境 (uv 會自動管理)
uv venv

# 啟動虛擬環境
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# 安裝所有依賴 (超快！)
uv pip install -e .

# 安裝開發工具
uv pip install -e ".[dev]"

# 驗證核心套件
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
```

**預期輸出**:

```
PyTorch: 2.1.2+cu121, CUDA: True
Transformers: 4.40.1
FAISS: 1.7.4
```

### 4.3 特殊套件安裝：PyTorch with CUDA

如果上述自動安裝的 PyTorch 沒有 CUDA 支援，手動安裝：

```bash
# 先移除舊版本
uv pip uninstall torch torchvision

# 安裝 CUDA 12.1 版本 (對應 RTX 5080)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 驗證
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# 應該輸出: True 12.1
```

### 4.4 安裝 FAISS GPU 版本 (如果失敗)

```bash
# 如果 faiss-gpu 安裝失敗，使用 conda 安裝
conda install -c pytorch -c nvidia faiss-gpu=1.7.4

# 或從源碼編譯 (進階)
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release .
make -C build -j
cd build/faiss/python && pip install .
```

### 4.5 生成依賴鎖定文件

```bash
# uv 自動生成 uv.lock (類似 poetry.lock)
# 這確保了環境的完全可重現性

# 查看依賴樹
uv pip tree

# 導出 requirements.txt (供不使用 uv 的人)
uv pip freeze > requirements.txt
```

---

## 5) 資料集下載與處理

### 5.1 下載腳本準備

建立資料集下載腳本：

```bash
cat > scripts/download_coco.sh << 'SCRIPT'
#!/bin/bash
# MS-COCO 2014 資料集下載腳本

set -e  # 遇到錯誤立即停止

# 設定變數
DATA_DIR="./data"
COCO_DIR="${DATA_DIR}/coco"
ANNO_DIR="${COCO_DIR}/annotations"
IMG_DIR="${COCO_DIR}/images"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MS-COCO 2014 Dataset Download Script${NC}"
echo -e "${GREEN}========================================${NC}"

# 建立目錄
mkdir -p ${ANNO_DIR}
mkdir -p ${IMG_DIR}/{train2014,val2014}

# COCO 下載 URLs
ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
TRAIN_URL="http://images.cocodataset.org/zips/train2014.zip"
VAL_URL="http://images.cocodataset.org/zips/val2014.zip"

# 函數：下載並解壓
download_and_extract() {
    local url=$1
    local output_dir=$2
    local filename=$(basename ${url})
    
    echo -e "${YELLOW}正在下載: ${filename}${NC}"
    
    # 下載 (使用 wget 或 curl)
    if command -v wget &> /dev/null; then
        wget -c ${url} -P ${output_dir}
    elif command -v curl &> /dev/null; then
        curl -C - -o ${output_dir}/${filename} ${url}
    else
        echo -e "${RED}錯誤: 請安裝 wget 或 curl${NC}"
        exit 1
    fi
    
    # 解壓
    echo -e "${YELLOW}正在解壓: ${filename}${NC}"
    unzip -q ${output_dir}/${filename} -d ${output_dir}
    
    # 刪除壓縮檔以節省空間 (可選)
    read -p "刪除壓縮檔 ${filename}? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm ${output_dir}/${filename}
        echo -e "${GREEN}已刪除 ${filename}${NC}"
    fi
}

# 1. 下載標註
echo -e "\n${GREEN}[1/3] 下載標註檔案...${NC}"
if [ ! -f "${ANNO_DIR}/instances_train2014.json" ]; then
    download_and_extract ${ANNO_URL} ${DATA_DIR}
    mv ${DATA_DIR}/annotations/* ${ANNO_DIR}/
    rmdir ${DATA_DIR}/annotations
else
    echo -e "${YELLOW}標註檔案已存在，跳過下載${NC}"
fi

# 2. 下載訓練影像
echo -e "\n${GREEN}[2/3] 下載訓練影像 (~13GB, 需時較久)...${NC}"
if [ ! -d "${IMG_DIR}/train2014" ] || [ -z "$(ls -A ${IMG_DIR}/train2014)" ]; then
    download_and_extract ${TRAIN_URL} ${IMG_DIR}
else
    echo -e "${YELLOW}訓練影像已存在，跳過下載${NC}"
fi

# 3. 下載驗證影像
echo -e "\n${GREEN}[3/3] 下載驗證影像 (~6GB)...${NC}"
if [ ! -d "${IMG_DIR}/val2014" ] || [ -z "$(ls -A ${IMG_DIR}/val2014)" ]; then
    download_and_extract ${VAL_URL} ${IMG_DIR}
else
    echo -e "${YELLOW}驗證影像已存在，跳過下載${NC}"
fi

# 驗證下載
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}驗證資料集完整性...${NC}"
echo -e "${GREEN}========================================${NC}"

# 計數檔案
train_count=$(find ${IMG_DIR}/train2014 -name "*.jpg" | wc -l)
val_count=$(find ${IMG_DIR}/val2014 -name "*.jpg" | wc -l)

echo -e "訓練影像數量: ${GREEN}${train_count}${NC} (預期: 82,783)"
echo -e "驗證影像數量: ${GREEN}${val_count}${NC} (預期: 40,504)"

if [ -f "${ANNO_DIR}/instances_train2014.json" ]; then
    echo -e "標註檔案: ${GREEN}✓${NC}"
else
    echo -e "標註檔案: ${RED}✗${NC}"
fi

# 顯示磁碟使用量
echo -e "\n磁碟使用量:"
du -sh ${COCO_DIR}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}下載完成！${NC}"
echo -e "${GREEN}========================================${NC}"
SCRIPT

# 賦予執行權限
chmod +x scripts/download_coco.sh
```

### 5.2 執行下載

```bash
# 開始下載 (需時約 1-3 小時，視網速而定)
./scripts/download_coco.sh

# 如果你在台灣/亞洲，可以使用鏡像站加速：
# 編輯腳本，將 URL 改為：
# TRAIN_URL="http://msvocds.blob.core.windows.net/coco2014/train2014.zip"
# VAL_URL="http://msvocds.blob.core.windows.net/coco2014/val2014.zip"
```

**預期目錄結構**:

```
data/coco/
├── annotations/
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   └── ...
└── images/
    ├── train2014/
    │   ├── COCO_train2014_000000000009.jpg
    │   ├── COCO_train2014_000000000025.jpg
    │   └── ... (82,783 張)
    └── val2014/
        ├── COCO_val2014_000000000042.jpg
        └── ... (40,504 張)
```

### 5.3 下載 Karpathy Split

```bash
# 建立下載 Karpathy split 的 Python 腳本
cat > scripts/download_karpathy_split.py << 'PYTHON'
#!/usr/bin/env python3
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
PYTHON

chmod +x scripts/download_karpathy_split.py

# 執行
python scripts/download_karpathy_split.py
```

### 5.4 預處理腳本（建立索引）

```bash
# 建立資料集索引腳本
cat > scripts/create_dataset_index.py << 'PYTHON'
#!/usr/bin/env python3
"""建立 COCO 資料集索引以加速訓練"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm

def create_index(data_dir, split="train2014"):
    """建立影像到標註的快速索引"""
    
    data_dir = Path(data_dir)
    anno_file = data_dir / "annotations" / f"instances_{split}.json"
    caption_file = data_dir / "annotations" / f"captions_{split}.json"
    
    print(f"正在處理 {split}...")
    
    # 載入 COCO API
    coco = COCO(anno_file)
    coco_caps = COCO(caption_file)
    
    # 建立索引
    index = {
        "images": {},
        "categories": {},
    }
    
    # 1. 類別資訊
    for cat_id, cat_info in coco.cats.items():
        index["categories"][cat_id] = {
            "id": cat_id,
            "name": cat_info["name"],
            "supercategory": cat_info["supercategory"]
        }
    
    # 2. 影像資訊
    for img_id in tqdm(coco.imgs.keys(), desc="Processing images"):
        img_info = coco.imgs[img_id]
        
        # 獲取該影像的所有標註
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # 提取類別 (multi-hot)
        categories = set([ann["category_id"] for ann in anns])
        
        # 獲取 captions
        cap_ids = coco_caps.getAnnIds(imgIds=img_id)
        caps = coco_caps.loadAnns(cap_ids)
        captions = [cap["caption"] for cap in caps]
        
        index["images"][img_id] = {
            "file_name": img_info["file_name"],
            "width": img_info["width"],
            "height": img_info["height"],
            "categories": sorted(list(categories)),  # 物件類別列表
            "captions": captions,  # 5 個 captions
        }
    
    # 儲存索引
    output_file = data_dir / f"index_{split}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(index, f)
    
    print(f"✓ 索引已儲存: {output_file}")
    print(f"  - 影像數量: {len(index['images'])}")
    print(f"  - 類別數量: {len(index['categories'])}")
    
    return index

if __name__ == "__main__":
    data_dir = Path("./data/coco")
    
    # 處理訓練集與驗證集
    for split in ["train2014", "val2014"]:
        create_index(data_dir, split)
    
    print("\n✓ 所有索引建立完成！")
PYTHON

chmod +x scripts/create_dataset_index.py

# 執行（需安裝 pycocotools）
python scripts/create_dataset_index.py
```

### 5.5 資料集統計分析

```bash
# 建立統計腳本
cat > scripts/analyze_dataset.py << 'PYTHON'
#!/usr/bin/env python3
"""分析 COCO 資料集統計資訊"""

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
PYTHON

chmod +x scripts/analyze_dataset.py

# 執行分析
python scripts/analyze_dataset.py
```

---

## 6) 驗證環境

### 6.1 建立驗證腳本

```bash
cat > scripts/verify_setup.py << 'PYTHON'
#!/usr/bin/env python3
"""驗證實驗環境設置"""

import sys
from pathlib import Path

def check_python():
    """檢查 Python 版本"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 10, "需要 Python 3.10+"

def check_cuda():
    """檢查 CUDA 可用性"""
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA 不可用 (將使用 CPU，速度會很慢)")
    return cuda_available

def check_packages():
    """檢查關鍵套件"""
    packages = {
        "torch": "2.1.0",
        "transformers": "4.40.0",
        "faiss": "1.7.0",
        "pycocotools": "2.0.0",
    }
    
    for pkg, min_version in packages.items():
        try:
            mod = __import__(pkg)
            version = mod.__version__ if hasattr(mod, "__version__") else "unknown"
            print(f"✓ {pkg:20s} {version}")
        except ImportError:
            print(f"✗ {pkg:20s} 未安裝")
            return False
    
    return True

def check_dataset():
    """檢查資料集"""
    data_dir = Path("./data/coco")
    
    checks = [
        data_dir / "images/train2014",
        data_dir / "images/val2014",
        data_dir / "annotations/instances_train2014.json",
        data_dir / "annotations/captions_train2014.json",
        data_dir / "index_train2014.pkl",
    ]
    
    all_exist = True
    for path in checks:
        if path.exists():
            if path.is_dir():
                n_files = len(list(path.glob("*.jpg")))
                print(f"✓ {path} ({n_files:,} 張影像)")
            else:
                size_mb = path.stat().st_size / 1e6
                print(f"✓ {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {path} 不存在")
            all_exist = False
    
    return all_exist

def check_siglip2():
    """檢查 SigLIP2 模型載入"""
    from transformers import Siglip2Model, Siglip2Processor
    
    try:
        print("\n正在測試 SigLIP2 模型載入...")
        model_name = "google/siglip2-base-patch16-256"
        processor = Siglip2Processor.from_pretrained(model_name)
        model = Siglip2Model.from_pretrained(model_name)
        print(f"✓ SigLIP2 模型載入成功")
        print(f"  參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"✗ SigLIP2 模型載入失敗: {e}")
        return False

def main():
    print("="*60)
    print("環境驗證")
    print("="*60)
    
    print("\n[1/5] 檢查 Python 版本...")
    check_python()
    
    print("\n[2/5] 檢查 CUDA...")
    cuda_ok = check_cuda()
    
    print("\n[3/5] 檢查 Python 套件...")
    pkg_ok = check_packages()
    
    print("\n[4/5] 檢查資料集...")
    data_ok = check_dataset()
    
    print("\n[5/5] 檢查 SigLIP2 模型...")
    model_ok = check_siglip2()
    
    print("\n" + "="*60)
    if all([cuda_ok, pkg_ok, data_ok, model_ok]):
        print("✅ 環境設置完成！可以開始實驗了。")
    else:
        print("⚠️  部分檢查失敗，請修正後再試。")
    print("="*60)

if __name__ == "__main__":
    main()
PYTHON

chmod +x scripts/verify_setup.py

# 執行驗證
python scripts/verify_setup.py
```

### 6.2 測試 SigLIP2 前向傳播

```bash
cat > scripts/test_siglip2.py << 'PYTHON'
#!/usr/bin/env python3
"""測試 SigLIP2 模型前向傳播"""

import torch
from transformers import Siglip2Model, Siglip2Processor
from PIL import Image
import requests
from io import BytesIO

# 載入模型
print("載入 SigLIP2 模型...")
model_name = "google/siglip2-base-patch16-256"
processor = Siglip2Processor.from_pretrained(model_name)
model = Siglip2Model.from_pretrained(model_name)

# 移到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"模型已載入到 {device}")

# 下載測試影像
print("\n下載測試影像...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# 準備輸入
text = "Two cats sleeping on a couch"
inputs = processor(
    text=[text],
    images=image,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 前向傳播
print("\n執行前向傳播...")
with torch.no_grad():
    outputs = model(**inputs)

# 檢查輸出
image_embeds = outputs.image_embeds  # (1, D)
text_embeds = outputs.text_embeds    # (1, D)

print(f"✓ 影像 embedding shape: {image_embeds.shape}")
print(f"✓ 文字 embedding shape: {text_embeds.shape}")

# 計算相似度
similarity = torch.cosine_similarity(image_embeds, text_embeds)
print(f"✓ 圖文相似度: {similarity.item():.4f}")

print("\n✅ SigLIP2 測試通過！")
PYTHON

chmod +x scripts/test_siglip2.py
python scripts/test_siglip2.py
```

---

## 7) 常見問題排除

### 7.1 uv 安裝失敗

**問題**: `curl: command not found`

**解決**:

```bash
# Ubuntu/Debian
sudo apt install curl

# 或直接用 pip
pip install uv
```

### 7.2 CUDA 不可用

**問題**: `torch.cuda.is_available()` 返回 `False`

**檢查清單**:

```bash
# 1. 檢查 NVIDIA 驅動
nvidia-smi

# 2. 檢查 PyTorch CUDA 版本是否匹配
python -c "import torch; print(torch.version.cuda)"

# 3. 重新安裝正確的 PyTorch 版本
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 7.3 資料集下載速度慢

**解決方案 1: 使用鏡像站**

```bash
# 編輯 scripts/download_coco.sh，改用 Azure 鏡像
TRAIN_URL="http://msvocds.blob.core.windows.net/coco2014/train2014.zip"
VAL_URL="http://msvocds.blob.core.windows.net/coco2014/val2014.zip"
```

**解決方案 2: 使用 aria2c 多線程下載**

```bash
# 安裝 aria2c
sudo apt install aria2

# 多線程下載
aria2c -x 16 -s 16 http://images.cocodataset.org/zips/train2014.zip
```

**解決方案 3: 使用學術網路 VPN**

- 許多大學提供 VPN，連線後下載速度會大幅提升

### 7.4 FAISS GPU 版本編譯失敗

**方案 A: 使用 Conda 安裝**

```bash
# 即使在 uv 環境中，也可以用 conda 裝 faiss
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

**方案 B: 先用 CPU 版本**

```bash
# 暫時使用 CPU 版本（速度較慢但能用）
uv pip install faiss-cpu

# 等需要大規模檢索時再換回 GPU 版本
```

### 7.5 記憶體不足（RAM）

**問題**: 處理資料集時系統記憶體不足

**解決**:

```bash
# 1. 關閉不必要的程式
# 2. 使用 swap (臨時)
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 3. 或分批處理資料（修改腳本）
```

### 7.6 pycocotools 安裝失敗

**問題**: `error: Microsoft Visual C++ 14.0 is required`（Windows）

**解決**:

```bash
# 安裝 Visual Studio Build Tools
# 或使用預編譯輪子
uv pip install pycocotools-windows
```

**Linux**:

```bash
# 安裝依賴
sudo apt install python3-dev build-essential
uv pip install pycocotools
```

---

## 8) 下一步

環境設置完成後，可以開始：

### 8.1 建立配置檔案

```bash
# 建立第一個實驗配置
mkdir -p configs/experiments
cat > configs/experiments/baseline.yaml << 'YAML'
# Baseline 實驗配置
name: baseline_siglip2_hash64_knn20

model:
  siglip2_variant: google/siglip2-base-patch16-256
  max_num_patches: 256
  freeze_towers: true
  hash_bits: 64

training:
  batch_size: 64
  num_epochs: 30
  lr: 3e-4

knn:
  K: 20
  tau: 0.07
YAML
```

### 8.2 開始開發

```bash
# 建立主要程式碼檔案
touch src/{__init__,model,dataset,train,evaluate}.py

# 或參考完整計畫文件中的程式碼範例
```

### 8.3 啟動實驗追蹤

```bash
# 登入 Weights & Biases (可選)
wandb login

# 或使用 TensorBoard
tensorboard --logdir experiments/
```

---

## 總結

你現在已經完成：

- ✅ 安裝了超快速的 `uv` 套件管理器
- ✅ 建立了隔離的 Python 虛擬環境
- ✅ 安裝了所有必要的依賴套件（PyTorch, Transformers, FAISS 等）
- ✅ 下載了 MS-COCO 2014 資料集（~20GB）
- ✅ 建立了資料集索引以加速訓練
- ✅ 驗證了 SigLIP2 模型可正常運作
- ✅ 建立了完整的專案目錄結構

**環境設置檢查表**:

```
[O] uv 已安裝且可用
[O] Python 3.10+ 虛擬環境已建立
[O] PyTorch with CUDA 可正常運作
[O] Transformers 套件已安裝
[O] FAISS-GPU 已安裝 (Use CPU Version instead)
[-] COCO 2014 訓練/驗證影像已下載
□ COCO 標註與 captions 已下載
□ Karpathy split 已下載
□ 資料集索引已建立
□ SigLIP2 模型可載入並前向傳播
```

**下一步建議**:

1. 📖 閱讀完整實驗計畫（`siglip2_multimodal_hash_knn_complete_plan.md`）
2. 💻 實作資料載入器（`src/dataset.py`）
3. 🏗️ 實作模型架構（`src/model.py`）
4. 🚀 開始訓練！

祝實驗順利！有任何問題隨時詢問。
PYTHON

```

恭喜你完成環境設置！🎉

# 硬體配置優化指南 - RTX 5080 16GB 專用

> **你的硬體規格**:
> - GPU: NVIDIA GeForce RTX 5080 (16GB VRAM)
> - CPU: 32 核心
> - RAM: 42 GB
> - CUDA: 13.0
> - Driver: 580.126.09

---

## ⚠️ 重要提醒：VRAM 限制

你的 GPU 只有 **16GB VRAM**（而非原計畫假設的 24GB），因此需要調整以下參數：

### 原始配置 vs 優化配置對比

| 參數 | 原始 (24GB) | 優化 (16GB) | 說明 |
| ------ | ------------ | ------------ | ------ |
| `batch_size` | 64 | **32** | 減半以節省記憶體 |
| `max_num_patches` | 256 | **256** (保持) | 可嘗試但需監控 |
| `mixed_precision` | 建議 | **必須** | FP16 可節省 40% VRAM |
| `gradient_accumulation` | 可選 | **推薦 2-4** | 模擬大 batch size |
| `freeze_towers` | true | **true** | 必須凍結以節省記憶體 |

---

## 1) 優化後的 pyproject.toml

```toml
[project]
name = "siglip2-multimodal-hash"
version = "0.1.0"
description = "Multimodal image-text multi-label classification using SigLIP2, hashing, and KNN"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    # Deep Learning Framework (CUDA 13.0 相容)
    "torch>=2.5.0",  # 支援 CUDA 13.0
    "torchvision>=0.20.0",
    
    # Transformers & Vision-Language Models
    "transformers>=4.47.0",  # 確保支援 SigLIP2
    "accelerate>=1.2.0",
    
    # Computer Vision
    "opencv-python>=4.10.0",
    "pillow>=11.0.0",
    "albumentations>=1.4.0",
    
    # Data Processing
    "numpy>=1.26.0,<2.0.0",  # 確保相容性
    "pandas>=2.2.0",
    "pycocotools>=2.0.7",
    
    # Similarity Search & Indexing
    "faiss-gpu>=1.9.0",  # CUDA 13.0 相容版本
    
    # Metrics & Evaluation
    "scikit-learn>=1.5.0",
    "scipy>=1.14.0",
    
    # Visualization
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "plotly>=5.24.0",
    
    # Configuration & Logging
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb>=0.18.0",
    "tensorboard>=2.18.0",
    
    # Utilities
    "tqdm>=4.67.0",
    "rich>=13.9.0",
    "python-dotenv>=1.0.0",
    
    # Development Tools
    "ipython>=8.29.0",
    "jupyter>=1.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "black>=24.10.0",
    "ruff>=0.8.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 2) 針對 16GB VRAM 的訓練配置

### configs/hardware/rtx5080_16gb.yaml

```yaml
# RTX 5080 16GB 硬體優化配置
hardware:
  device: cuda
  gpu_model: "RTX 5080"
  vram_gb: 16
  cuda_version: "13.0"
  
  # 記憶體優化
  memory_optimization:
    mixed_precision: true  # 必須啟用！
    gradient_checkpointing: true  # 節省 30-40% 記憶體
    empty_cache_steps: 100  # 定期清理 GPU 快取
    pin_memory: true
    non_blocking: true

# 模型配置（針對 16GB 優化）
model:
  siglip2_variant: "google/siglip2-base-patch16-256"  # 不要用 large！
  max_num_patches: 256  # 保守設定
  text_max_length: 64
  freeze_towers: true  # 必須凍結！
  
  fusion:
    mlp_dims: [1024, 512]  # 保持原設計
    dropout: 0.1
  
  hash:
    bits: 64  # 可嘗試 128，但需監控記憶體

# 訓練配置
training:
  # 批次大小（關鍵！）
  batch_size: 32  # 原本 64 太大
  gradient_accumulation_steps: 2  # 模擬 batch_size=64
  
  # 有效 batch size = 32 * 2 = 64
  effective_batch_size: 64
  
  # Epoch 與學習率
  num_epochs: 30
  warmup_epochs: 2
  
  # DataLoader 優化（利用你的 32 核心 CPU！）
  num_workers: 16  # 你有 32 核心，可以多用點
  prefetch_factor: 3
  persistent_workers: true
  
  # 記憶體管理
  gradient_clip_norm: 1.0
  max_grad_norm: 1.0

# Optimizer（針對小 batch 調整）
optimizer:
  type: adamw
  lr: 2e-4  # 比原本的 3e-4 略小（因為 effective batch size 一樣）
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8

# Scheduler
scheduler:
  type: cosine_with_warmup
  warmup_ratio: 0.1
  min_lr: 1e-6

# 監控與檢查點
monitoring:
  log_every_n_steps: 50
  val_every_n_epochs: 1
  save_top_k: 3
  early_stopping_patience: 5
  
  # VRAM 監控（重要！）
  log_gpu_memory: true
  alert_vram_threshold: 14.5  # 超過 14.5GB 發出警告

# KNN 推論配置
knn:
  K: 20
  tau: 0.07
  batch_size: 64  # 推論時可以大一點
```

---

## 3) 記憶體佔用估算表（16GB VRAM）

### Base 配置下的記憶體分配

| 組件 | 記憶體佔用 | 說明 |
| ------ | ----------- | ------ |
| **SigLIP2-base (凍結)** | ~2.5 GB | 僅 forward pass |
| **Fusion MLP** | ~0.3 GB | 可訓練 |
| **Hash Layer** | ~0.1 GB | 可訓練 |
| **Classifier Head** | ~0.05 GB | 可訓練 |
| **Optimizer States** | ~1.2 GB | AdamW (2x 參數量) |
| **Batch Data (32)** | ~4.0 GB | Images + embeddings |
| **Gradients** | ~0.5 GB | 僅可訓練部分 |
| **CUDA Kernels** | ~0.5 GB | PyTorch overhead |
| **預留緩衝** | ~1.0 GB | 安全邊界 |
| **總計** | **~10.2 GB** | ✅ 在 16GB 內安全 |

### ⚠️ 不要嘗試的配置

| 配置 | 預估 VRAM | 結果 |
| ------ | ----------- | ------ |
| batch_size=64 | ~16.5 GB | ❌ OOM (記憶體溢出) |
| SigLIP2-large | ~18 GB | ❌ OOM |
| max_patches=512 | ~14 GB | ⚠️ 可能 OOM |
| 不凍結 towers | ~22 GB | ❌ OOM |

---

## 4) 安裝 PyTorch for CUDA 13.0

你的系統有 **CUDA 13.0**（非常新！），需要確保 PyTorch 相容：

```bash
# 啟動虛擬環境
cd ~/projects/siglip2-multimodal-hash
source .venv/bin/activate

# 安裝最新版 PyTorch (支援 CUDA 13.0)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 驗證
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

**預期輸出**:

```
PyTorch: 2.5.1+cu130
CUDA available: True
CUDA version: 13.0
GPU: NVIDIA GeForce RTX 5080
VRAM: 16.3 GB
```

---

## 5) FAISS GPU for CUDA 13.0

FAISS 可能還沒有官方的 CUDA 13.0 輪子，使用以下方法：

### 方法 A: 使用 Conda (推薦)

```bash
# 即使在 uv 環境中，也可以用 conda 裝 FAISS
conda install -c pytorch -c nvidia faiss-gpu

# 驗證
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
```

### 方法 B: 暫時使用 CPU 版本

```bash
# 先用 CPU 版本開發
uv pip install faiss-cpu

# 等正式訓練時再換
```

### 方法 C: 從源碼編譯 (進階)

```bash
# 安裝依賴
sudo apt-get install cmake libopenblas-dev

# 下載並編譯
git clone https://github.com/facebookresearch/faiss.git
cd faiss

# 配置（針對 CUDA 13.0）
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  .

# 編譯（使用你的 32 核心！）
make -C build -j32

# 安裝
cd build/faiss/python
pip install .
```

**註**: RTX 5080 的 compute capability 是 **8.9**（Ada Lovelace 架構）

---

## 6) 訓練腳本範例（含記憶體監控）

建立一個訓練腳本，加入 VRAM 監控：

```python
#!/usr/bin/env python3
# scripts/train_with_memory_monitor.py

import torch
import psutil
from torch.cuda.amp import autocast, GradScaler

def get_gpu_memory_info():
    """獲取 GPU 記憶體使用資訊"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "free_gb": 16.0 - reserved  # 你的 GPU 總記憶體
        }
    return None

def train_epoch_optimized(model, dataloader, optimizer, scheduler, config):
    """優化的訓練迴圈（針對 16GB VRAM）"""
    
    model.train()
    scaler = GradScaler()  # 混合精度
    
    total_loss = 0
    accumulation_steps = config.training.gradient_accumulation_steps
    
    for batch_idx, batch in enumerate(dataloader):
        # 移到 GPU
        images = batch['images'].to('cuda', non_blocking=True)
        texts = batch['texts'].to('cuda', non_blocking=True)
        labels = batch['labels'].to('cuda', non_blocking=True)
        
        # 混合精度前向傳播
        with autocast():
            outputs = model(images, texts, return_components=True)
            loss = compute_loss(outputs, labels, config)
            loss = loss / accumulation_steps  # 梯度累積
        
        # 反向傳播
        scaler.scale(loss).backward()
        
        # 梯度累積
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            
            # 更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # 定期監控記憶體
        if batch_idx % 100 == 0:
            mem_info = get_gpu_memory_info()
            print(f"Batch {batch_idx}: GPU Memory: {mem_info['allocated_gb']:.2f}GB / 16GB")
            
            # 警告機制
            if mem_info['allocated_gb'] > 14.5:
                print("⚠️  警告：GPU 記憶體使用接近上限！")
        
        # 定期清理快取
        if batch_idx % config.hardware.memory_optimization.empty_cache_steps == 0:
            torch.cuda.empty_cache()
    
    scheduler.step()
    return total_loss / len(dataloader)

def compute_loss(outputs, labels, config):
    """損失計算（與原計畫相同）"""
    logits = outputs['logits']
    h = outputs['h']
    d_img = outputs['d_img']
    d_txt = outputs['d_txt']
    
    # BCE Loss
    loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    
    # Cosine Loss
    loss_cos = 1 - torch.nn.functional.cosine_similarity(d_img, d_txt, dim=1).mean()
    
    # Hash Regularization
    loss_hash = hash_regularization(h, config)
    
    # 組合
    total_loss = (
        config.loss.bce_weight * loss_bce +
        config.loss.cosine_weight * loss_cos +
        config.loss.hash_weight * loss_hash
    )
    
    return total_loss

def hash_regularization(h, config):
    """Hash 正則化（與原計畫相同）"""
    # Quantization
    loss_quant = torch.mean((torch.abs(h) - 1) ** 2)
    
    # Balance
    bit_mean = torch.mean(h, dim=0)
    loss_balance = torch.mean(bit_mean ** 2)
    
    # Decorrelation
    h_centered = h - torch.mean(h, dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / h.size(0)
    loss_decorr = (torch.sum(cov ** 2) - torch.trace(cov ** 2)) / (h.size(1) ** 2)
    
    return loss_quant + config.loss.hash_reg.lambda_balance * loss_balance + \
           config.loss.hash_reg.lambda_decorr * loss_decorr

if __name__ == "__main__":
    # 測試記憶體監控
    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        print("GPU 記憶體狀態:")
        print(f"  已分配: {mem_info['allocated_gb']:.2f} GB")
        print(f"  已保留: {mem_info['reserved_gb']:.2f} GB")
        print(f"  可用: {mem_info['free_gb']:.2f} GB")
```

---

## 7) 記憶體不足時的緊急措施

如果訓練時還是遇到 OOM，依序嘗試：

### 步驟 1: 降低 batch size

```yaml
training:
  batch_size: 16  # 從 32 降到 16
  gradient_accumulation_steps: 4  # 調整到 4
```

### 步驟 2: 降低解析度

```yaml
model:
  max_num_patches: 196  # 從 256 降到 196
```

### 步驟 3: 啟用更激進的優化

```python
# 在模型初始化時
model.model.gradient_checkpointing_enable()

# 或使用 torch 2.0 的 compile（如果支援）
model = torch.compile(model, mode="reduce-overhead")
```

### 步驟 4: 清理不必要的張量

```python
# 在訓練迴圈中
del outputs, loss
torch.cuda.empty_cache()
```

---

## 8) 預期訓練速度（你的硬體）

基於你的配置（RTX 5080 + 32 核心 CPU）：

| 配置 | 速度 (iter/s) | 每 Epoch 時間 | 備註 |
| ------ | -------------- | -------------- | ------ |
| batch=32, patches=256 | ~1.8 | ~35 分鐘 | 推薦配置 |
| batch=16, patches=256 | ~2.5 | ~50 分鐘 | OOM 備案 |
| batch=32, patches=196 | ~2.2 | ~30 分鐘 | 速度優先 |

**完整訓練時間估算**:

- 30 epochs × 35 分鐘 = **17.5 小時**
- 建議分多次訓練（每次 10 epochs），定期檢查

---

## 9) 完整的記憶體優化檢查清單

在開始訓練前，確認以下所有項目：

```bash
# 檢查清單腳本
cat > scripts/check_memory_config.py << 'EOF'
#!/usr/bin/env python3
import torch
import yaml

def check_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    checks = {
        "✅ Mixed precision enabled": config['hardware']['memory_optimization']['mixed_precision'],
        "✅ Gradient checkpointing enabled": config['hardware']['memory_optimization']['gradient_checkpointing'],
        "✅ Batch size <= 32": config['training']['batch_size'] <= 32,
        "✅ Towers frozen": config['model']['freeze_towers'],
        "✅ Not using SigLIP-large": 'base' in config['model']['siglip2_variant'],
        "✅ Gradient accumulation configured": config['training']['gradient_accumulation_steps'] >= 2,
    }
    
    print("記憶體優化配置檢查:")
    for check, passed in checks.items():
        print(f"  {check if passed else check.replace('✅', '❌')}")
    
    if all(checks.values()):
        print("\n✅ 所有檢查通過！可以安全訓練。")
    else:
        print("\n⚠️  部分檢查失敗，建議修正後再訓練。")

if __name__ == "__main__":
    check_config("configs/hardware/rtx5080_16gb.yaml")
EOF

chmod +x scripts/check_memory_config.py
```

---

## 總結

### ✅ 針對你的硬體的關鍵調整

1. **Batch size**: 64 → **32**（必須）
2. **混合精度**: 建議 → **必須啟用**
3. **梯度累積**: 可選 → **推薦 2-4 步**
4. **凍結 towers**: true → **必須為 true**
5. **記憶體監控**: 加入 VRAM 追蹤與警告

### 🎯 你的優勢

- ✅ **32 核心 CPU** - 可以用更多 DataLoader workers
- ✅ **42GB RAM** - 資料預處理完全無壓力
- ✅ **最新 CUDA 13.0** - 支援最新優化
- ✅ **1TB+ 儲存空間** - 可以儲存大量實驗結果

### 📊 預期效能

- 訓練速度: **~1.8 iter/s** (batch_size=32)
- 每 epoch: **~35 分鐘**
- 完整訓練 (30 epochs): **~17.5 小時**

下一步就是按照原本的環境設置教學，但使用這份**硬體優化配置**來進行訓練！

祝實驗順利！🚀
