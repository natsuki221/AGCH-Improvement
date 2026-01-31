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
mkdir -p ${IMG_DIR}

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
    download_and_extract ${ANNO_URL} ${COCO_DIR}
    echo -e "${GREEN}標註檔案下載完成${NC}"
else
    echo -e "${YELLOW}標註檔案已存在，跳過下載${NC}"
fi

# 2. 下載訓練影像
echo -e "\n${GREEN}[2/3] 下載訓練影像 (~13GB, 需時較久)...${NC}"
if [ ! -d "${IMG_DIR}/train2014" ] || [ -z "$(ls -A ${IMG_DIR}/train2014 2>/dev/null)" ]; then
    download_and_extract ${TRAIN_URL} ${IMG_DIR}
else
    echo -e "${YELLOW}訓練影像已存在，跳過下載${NC}"
fi

# 3. 下載驗證影像
echo -e "\n${GREEN}[3/3] 下載驗證影像 (~6GB)...${NC}"
if [ ! -d "${IMG_DIR}/val2014" ] || [ -z "$(ls -A ${IMG_DIR}/val2014 2>/dev/null)" ]; then
    download_and_extract ${VAL_URL} ${IMG_DIR}
else
    echo -e "${YELLOW}驗證影像已存在，跳過下載${NC}"
fi

# 驗證下載
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}驗證資料集完整性...${NC}"
echo -e "${GREEN}========================================${NC}"

# 計數檔案
train_count=$(find ${IMG_DIR}/train2014 -name "*.jpg" 2>/dev/null | wc -l)
val_count=$(find ${IMG_DIR}/val2014 -name "*.jpg" 2>/dev/null | wc -l)

echo -e "訓練影像數量: ${GREEN}${train_count}${NC} (預期: 82,783)"
echo -e "驗證影像數量: ${GREEN}${val_count}${NC} (預期: 40,504)"

# 檢查標註檔案
echo -e "\n標註檔案檢查:"
for anno in instances_train2014.json instances_val2014.json \
            captions_train2014.json captions_val2014.json \
            person_keypoints_train2014.json person_keypoints_val2014.json; do
    if [ -f "${ANNO_DIR}/${anno}" ]; then
        echo -e "  ${anno}: ${GREEN}✓${NC}"
    else
        echo -e "  ${anno}: ${YELLOW}✗${NC}"
    fi
done

# 顯示磁碟使用量
echo -e "\n磁碟使用量:"
du -sh ${COCO_DIR}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}下載完成！${NC}"
echo -e "${GREEN}========================================${NC}"
