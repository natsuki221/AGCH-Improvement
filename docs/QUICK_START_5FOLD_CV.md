# 五折交叉驗證快速執行指南

> **這是簡化版指南，完整文檔請見 `siglip2_multimodal_hash_5fold_cv_plan_v2.3_AGCH.md`**

---

## 🚀 快速開始（5 步驟）

### 步驟 1: 生成 Fold 切分

```bash
cd ~/Documents/Coding/github.com/natsuki221/AGCH-Improvement

# 生成 5-fold split
python scripts/create_kfold_split.py

# 驗證輸出
cat data/coco/5fold_split.json | python -m json.tool | head -20
```

**預期輸出**:

```
開發集池總數: 118287
測試集總數: 5000 (已排除)
 - fold_0: Train=94,630, Val=23,657
 - fold_1: Train=94,630, Val=23,657
 ...
✓ 已儲存至: data/coco/5fold_split.json
```

---

### 步驟 2: 建立配置檔案

```bash
# 建立目錄
mkdir -p configs/experiments

# 創建 configs/experiments/cv_experiment.yaml
cat > configs/experiments/cv_experiment.yaml << 'EOF'
defaults:
  - /hardware/rtx5080_16gb

experiment:
  name: "cv_baseline"
  seed: 42

k_fold:
  enabled: true
  num_folds: 5
  current_fold: 0

training:
  batch_size: 32
  gradient_accumulation_steps: 2
  num_epochs: 20  # 從 30 降到 20
  warmup_epochs: 1
  early_stopping_patience: 3  # 從 5 降到 3
  val_every_n_epochs: 1

checkpointing:
  save_dir: "./outputs/checkpoints"
  save_top_k: 1  # 只儲存最佳
  save_optimizer: false  # 不儲存 optimizer（節省空間）
  filename_format: "best_model_mAP{val_mAP:.4f}.pth"

logging:
  use_wandb: true
  wandb_project: "siglip2-5fold-cv"
  wandb_entity: "natsuki221"
EOF
```

---

### 步驟 3: 測試單個 Fold（重要！）

```bash
# 試跑 1 個 epoch，確保一切正常
python scripts/train.py \
    --config-name cv_experiment \
    k_fold.enabled=true \
    k_fold.current_fold=0 \
    training.num_epochs=1

# 檢查輸出：
# ✓ GPU 記憶體應在 10-11 GB
# ✓ 每個 epoch 約 13-15 分鐘
# ✓ Loss 正常下降
```

**如果成功**，你會看到：

```
Epoch 1/1
Train - Loss: 0.4523 (BCE: 0.3821, Cos: 0.0512, Hash: 0.0190)
Val   - Loss: 0.4123, mAP: 0.3245, F1-Micro: 0.4567
✓ 儲存最佳模型
```

---

### 步驟 4: 執行完整 5-Fold

```bash
# 賦予執行權限
chmod +x scripts/run_5fold_cv.sh

# 使用 tmux（推薦，因為要跑很久）
tmux new -s cv_training

# 在 tmux 中執行
./scripts/run_5fold_cv.sh

# 分離 tmux: 按 Ctrl+B，然後按 D
# 重新連接: tmux attach -t cv_training
```

**預期時長**: 約 17-22 小時（取決於 early stopping）

---

### 步驟 5: 聚合結果

```bash
# 執行結果聚合
python scripts/aggregate_cv_results.py \
    --exp_prefix siglip2_cv_run1_fold

# 查看結果
cat outputs/checkpoints/cv_results_summary.json
```

**預期輸出**:

```
🏆 5-Fold Cross-Validation 最終結果 (mAP)
============================================================
Mean: 0.7193
Std:  0.0082
Min:  0.7098
Max:  0.7311
Median: 0.7189
============================================================

📝 論文報告格式:
   mAP: 0.72 ± 0.01
============================================================
```

---

## 📊 關鍵指標對比

| 項目 | 單次訓練 (v2.2) | 五折驗證 (v2.3) |
| ------ | ---------------- | ---------------- |
| **訓練次數** | 1 | **5** |
| **總時長** | ~17.5 小時 | **~20 小時** |
| **結果形式** | mAP: 0.72 | **mAP: 0.72 ± 0.01** |
| **論文價值** | 中等 | **高（頂會標準）** |
| **硬碟需求** | ~5 GB | **~4 GB**（精簡儲存） |

---

## ⚠️ 故障排除

### 問題 1: FileNotFoundError: 5fold_split.json

**解決**: 執行 `python scripts/create_kfold_split.py`

### 問題 2: OOM (記憶體不足)

**解決**: 降低 batch size

```yaml
# 在 cv_experiment.yaml 中
training:
  batch_size: 16  # 從 32 降到 16
  gradient_accumulation_steps: 4  # 從 2 增到 4
```

### 問題 3: 訓練速度慢

**檢查**:

```bash
nvidia-smi  # GPU 利用率應 > 80%
```

**解決**: 增加 workers

```yaml
dataloader:
  num_workers: 20  # 從 16 增到 20
```

---

## 📁 重要檔案位置

```
AGCH-Improvement/
├── data/coco/
│   └── 5fold_split.json          ← create_kfold_split.py 生成
│
├── configs/experiments/
│   └── cv_experiment.yaml         ← 手動創建
│
├── scripts/
│   ├── create_kfold_split.py      ← 完整文檔中提供
│   ├── run_5fold_cv.sh            ← 完整文檔中提供
│   ├── aggregate_cv_results.py    ← 完整文檔中提供
│   └── train.py                   ← 需要小幅修改
│
└── outputs/checkpoints/
    ├── siglip2_cv_run1_fold0/
    ├── siglip2_cv_run1_fold1/
    ├── ...
    └── cv_results_summary.json    ← 最終結果
```

---

## 🎯 論文報告範例

```
We evaluate our method using 5-fold cross-validation on the 
MS-COCO dataset (118,287 development images). The results 
demonstrate strong and stable performance:

Results (5-Fold CV):
  - mAP (macro): 0.72 ± 0.01
  - F1-score (micro): 0.74 ± 0.01
  - F1-score (macro): 0.65 ± 0.02

The low standard deviation (< 0.02) across all folds indicates 
robust generalization, validating the effectiveness of our 
approach.
```

---

## 📞 需要幫助？

詳細說明請參考：

- **完整計畫**: `siglip2_multimodal_hash_5fold_cv_plan_v2.3_AGCH.md`
- **章節 6**: 完整程式碼實作
- **章節 10**: 詳細故障排除

祝實驗順利！🚀
