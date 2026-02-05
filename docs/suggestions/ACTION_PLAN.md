# 🎯 下一步行動計畫清單

> **基於實驗 20260205-5FOLD-SIGLIP2 的分析**  
> **當前狀態**: 5-Fold CV 已完成，mAP: 0.6785 ± 0.012

---

## 🚨 緊急優先 (必須完成)

### ✅ Task 1: Test Set 最終評估
**預計時間**: 2 小時  
**重要性**: ⭐⭐⭐⭐⭐ (論文必需)

```bash
# 使用最佳 fold (Fold 1, mAP=0.6982) 在 test set 上評估
python scripts/test_on_holdout.py \
    --checkpoint outputs/checkpoints/siglip2_cv_run1_fold1/best_model.pth \
    --test_split test \
    --output_file outputs/test_results_fold1.json

# 預期結果: Test mAP ≈ 0.67-0.69
```

**檢查點**:
- [ ] Test set 載入正確 (5,000 張影像)
- [ ] 模型推論成功
- [ ] 生成完整評估報告
- [ ] 更新 EXPERIMENT_REPORT.md

---

### ✅ Task 2: 補充完整評估指標
**預計時間**: 1 小時  
**重要性**: ⭐⭐⭐⭐

```python
# 在 aggregate_cv_results.py 中加入
metrics = [
    'mAP',
    'f1_micro',
    'f1_macro', 
    'precision_macro',
    'recall_macro',
    'auc_roc_macro',
    'hamming_loss'
]
```

**檢查點**:
- [ ] 5-Fold CV 的所有指標統計
- [ ] Test set 的所有指標
- [ ] 生成論文格式表格

---

## 🎯 高優先 (強烈建議)

### ✅ Task 3: Per-class 分析
**預計時間**: 3 小時  
**重要性**: ⭐⭐⭐⭐

```python
# 建立腳本: scripts/analyze_per_class.py
def analyze_per_class_performance():
    """分析每個類別的 AP"""
    # 1. 計算 per-class AP
    # 2. 找出 Top 10 / Bottom 10
    # 3. 視覺化分布
    # 4. 分析類別頻率與效能的關係
```

**預期發現**:
- 常見類別 (person, car) 表現好 (AP > 0.80)
- 長尾類別 (toothbrush, hair drier) 表現差 (AP < 0.40)
- 中等頻率類別可能表現不穩定

**檢查點**:
- [ ] Per-class AP 表格
- [ ] 視覺化圖表 (bar chart)
- [ ] 加入實驗報告

---

### ✅ Task 4: Ensemble 預測
**預計時間**: 4 小時  
**重要性**: ⭐⭐⭐⭐

```python
# scripts/ensemble_predict.py
def ensemble_5fold_models():
    """組合 5 個 fold 的模型進行預測"""
    models = []
    for fold_idx in range(5):
        ckpt = torch.load(f'outputs/checkpoints/fold{fold_idx}/best_model.pth')
        model = load_model(ckpt)
        models.append(model)
    
    # 在 test set 上做 ensemble
    ensemble_probs = average_predictions(models, test_loader)
    return ensemble_probs
```

**預期效果**:
- Test mAP 提升 2-3% (0.68 → 0.70+)
- 穩定性進一步提升

**檢查點**:
- [ ] 5 個模型載入成功
- [ ] Ensemble 推論完成
- [ ] 對比單模型 vs ensemble

---

## 💡 中優先 (時間允許)

### ✅ Task 5: Fold 1 異常分析
**預計時間**: 2 小時  
**重要性**: ⭐⭐⭐

```python
# 分析為何 Fold 1 (69.82%) 明顯高於其他
def analyze_fold_distribution(fold_idx):
    # 1. 統計該 fold 的類別分布
    # 2. 計算樣本難度分布
    # 3. 對比其他 fold
    # 4. 視覺化差異
```

**可能發現**:
- Fold 1 的驗證集類別分布更均勻
- 或包含較少困難樣本

**解決方案**:
- 考慮使用 Stratified K-Fold
- 或報告時說明此差異

---

### ✅ Task 6: 錯誤分析 (Error Analysis)
**預計時間**: 3 小時  
**重要性**: ⭐⭐⭐

```python
# scripts/error_analysis.py
def analyze_errors(model, test_loader):
    # 1. 找出誤報最多的類別對 (dog → cat)
    # 2. 找出漏報最多的類別
    # 3. 視覺化失敗案例 (影像 + GT + Pred)
    # 4. 分析失敗模式
```

**產出**:
- False Positive 分析表
- False Negative 分析表  
- 失敗案例視覺化 (10-20 個)

---

## 🔬 低優先 (研究擴展)

### ✅ Task 7: Ablation Study
**預計時間**: 3-5 天  
**重要性**: ⭐⭐⭐⭐⭐ (論文必需)

#### 7.1 Hash Bits
```yaml
# configs/ablation/hash_bits.yaml
experiments:
  - hash_bits: 32
  - hash_bits: 64  # baseline
  - hash_bits: 128
```

#### 7.2 Fusion Strategy
```yaml
experiments:
  - fusion: "concat_only"
  - fusion: "hadamard"  # baseline
  - fusion: "hadamard_with_magnitude"
```

#### 7.3 KNN vs MLP
```yaml
experiments:
  - classifier: "mlp_only"
  - classifier: "knn_only"
  - classifier: "hybrid"  # baseline
```

**檢查點**:
- [ ] 每個 ablation 至少 3 個設定
- [ ] 每個設定執行 3 次 (報告 mean ± std)
- [ ] 視覺化對比圖

---

### ✅ Task 8: 優化實驗
**預計時間**: 2-3 天  
**重要性**: ⭐⭐⭐

#### 8.1 資料增強
```python
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3)
])
```

#### 8.2 學習率調度
```yaml
scheduler:
  type: "cosine_annealing_warm_restarts"
  T_0: 5
  T_mult: 2
```

#### 8.3 部分解凍
```python
# 解凍 SigLIP2 最後幾層
unfreeze_last_n_layers(model.siglip2, n=2)
```

---

## 📊 時間規劃

### Week 1 (當前週)
- [x] 完成 5-Fold CV  
- [ ] **Task 1**: Test Set 評估 (0.5 天)
- [ ] **Task 2**: 補充指標 (0.5 天)
- [ ] **Task 3**: Per-class 分析 (1 天)
- [ ] **Task 4**: Ensemble (1 天)

### Week 2
- [ ] **Task 5**: Fold 1 分析 (0.5 天)
- [ ] **Task 6**: 錯誤分析 (1 天)
- [ ] **Task 7**: Ablation Study (3 天)

### Week 3
- [ ] **Task 8**: 優化實驗 (2 天)
- [ ] 整理所有結果 (1 天)
- [ ] 撰寫論文草稿 (2 天)

---

## 📈 預期最終結果

完成所有任務後：

```
╔═══════════════════════════════════════════════╗
║           最終實驗結果                        ║
╠═══════════════════════════════════════════════╣
║  5-Fold CV mAP:     0.6785 ± 0.012           ║
║  Test Set mAP:      ~0.68                     ║
║  Ensemble mAP:      ~0.70 (+2.6%)             ║
║  w/ Augmentation:   ~0.71 (+3.8%)             ║
║  w/ Fine-tuning:    ~0.73 (+6.3%)             ║
║                                               ║
║  論文價值:          ⭐⭐⭐⭐⭐                  ║
║  投稿建議:          CVPR/ICCV/ECCV workshop   ║
║                     或 ACM MM main track      ║
╚═══════════════════════════════════════════════╝
```

---

## ✅ 完成檢查清單

### 必須完成 (論文發表)
- [ ] Test Set 最終評估
- [ ] 完整評估指標 (7+ metrics)
- [ ] Per-class 分析
- [ ] Ablation Study (3+ 項)
- [ ] 錯誤分析與視覺化

### 建議完成 (提升品質)
- [ ] Ensemble 預測
- [ ] 資料增強實驗
- [ ] 學習率調度優化
- [ ] Fold 分布分析

### 可選完成 (額外加分)
- [ ] 部分解凍實驗
- [ ] 不同 backbone 對比
- [ ] 推論速度分析
- [ ] 模型壓縮實驗

---

## 🎯 立即行動

**今天就做**:
```bash
# 1. Test Set 評估 (最重要！)
cd ~/Documents/Coding/github.com/natsuki221/AGCH-Improvement
python scripts/test_on_holdout.py \
    --checkpoint outputs/checkpoints/siglip2_cv_run1_fold1/best_model.pth

# 2. 補充指標
python scripts/aggregate_cv_results.py \
    --exp_prefix siglip2_cv_run1_fold \
    --include_all_metrics

# 3. 更新實驗報告
vim docs/EXPERIMENT_REPORT.md
```

**本週完成**:
- Task 1-4 (Test set + 指標 + Per-class + Ensemble)

**下週完成**:
- Task 5-7 (分析 + Ablation)

**最終目標**:
- 3 週內完成所有實驗
- 撰寫完整論文草稿
- 準備投稿材料

---

加油！你已經完成了最困難的部分（5-Fold CV），剩下的都是錦上添花！🚀
