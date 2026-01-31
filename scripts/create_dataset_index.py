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