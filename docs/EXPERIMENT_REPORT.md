# 實驗報告

> **專案**: AGCH-Improvement  
> **建立日期**: 2026-02-04  
> **用途**: 記錄所有模型訓練與驗證實驗的結果分析

---

<!-- 新的實驗記錄請加在此處，使用 --- 分隔 -->

### 實驗記錄 ID: 20260205-BASELINE-SIGLIP2-MLP

**日期**: 2026-02-05  
**類別**: Baseline 對比實驗  
**變更摘要**: 完成 SigLIP2-MLP Baseline 實驗，建立改進版效能對比基準。

#### 1. 實驗配置

| 項目 | Baseline (MLP) | 改進版 (AGCH) |
|:---|:---|:---|
| **模型架構** | SigLIP2 + MLP 分類器 | SigLIP2 + AGCH (方向/幅度分解 + Hadamard 融合) |
| **損失函數** | BCE only | BCE + Cosine + Hash 正則化 |
| **KNN 推論** | ❌ 無 | ✅ 有 |
| **Epochs** | 30 | 20 |
| **Batch Size** | 32 × 2 = 64 | 32 × 2 = 64 |

#### 2. 實驗結果：Baseline vs AGCH 改進版

| 指標 | Baseline (MLP) | 改進版 (AGCH) | 差異 |
|:---:|:---:|:---:|:---:|
| **mAP** | **0.8384** | 0.6787 | **+0.1597 (+23.5%)** |
| **AUC-Macro** | **0.9836** | 0.9613 | +0.0223 |
| **F1-Macro** | **0.7701** | 0.5522 | +0.2179 |
| **Precision-Macro** | **0.8890** | 0.7728 | +0.1162 |
| **Recall-Macro** | **0.6972** | 0.4755 | +0.2217 |
| **Hamming Loss** | **0.0129** | 0.0190 | -0.0061 (更低更好) |
| **LRAP** | **0.9275** | 0.8476 | +0.0799 |

> [!IMPORTANT]
> **Baseline 在所有指標上都大幅超越 AGCH 改進版！**  
> 這表示 AGCH 架構（方向/幅度分解 + Hash + KNN）在此設定下反而造成效能下降。

#### 3. 分析與結論

| 觀察 | 分析 |
|:---|:---|
| **Baseline mAP 高達 0.8384** | 簡單的 MLP concat 策略足以達到極高效能 |
| **改進版 mAP 僅 0.6787** | Hash 層與 KNN 推論可能帶來資訊損失 |
| **Baseline 用更多 Epochs (30 vs 20)** | 公平對比需統一 epochs 數量 |
| **Recall 差距最大 (+0.22)** | 改進版可能過度 precision-oriented |

#### 4. 後續建議

1. **重新評估 AGCH 架構**: 考慮是否應繼續使用 Hash + KNN 策略
2. **消融實驗**: 拆解各組件貢獻 (僅 Hash、僅 decomposition 等)
3. **Hyperparameter 調整**: 嘗試不同 hash bits、loss 權重
4. **統一 Epochs**: 以相同 epochs 進行公平對比

---

### 實驗記錄 ID: 20260205-5FOLD-SIGLIP2

**日期**: 2026-02-05  
**類別**: 5-Fold 交叉驗證 (Cross-Validation)  
**變更摘要**: 完成 SigLIP2 + AGCH 改進版的 5-fold 穩定性與效能驗證。

#### 1. 實驗配置
- **模型**: SigLIP2 (frozen towers) + AGCH
- **Fold 數量**: 5
- **訓練輪數**: 20 Epochs per fold
- **硬體**: RTX 5080 (16GB)

#### 2. 實驗結果 (5-Fold CV Summary)

| 指標 | Mean | Std (±) | 評語 |
|:---:|:---:|:---:|:---|
| **mAP** | **0.6787** | **0.0129** | 表現穩定，屬中上水準 |
| **AUC-Macro** | **0.9613** | **0.0009** | 極高，模型排序能力強 |
| **F1-Macro** | **0.5522** | **0.0220** | 召回率仍有提升空間 |

**各 Fold 詳細表現 (Epoch 18-19)**:

| Fold | mAP | AUC | F1-Macro |
|:---:|:---:|:---:|:---:|
| 0 | 0.6644 | 0.9606 | 0.5310 |
| 1 | 0.6982 | 0.9629 | 0.5867 |
| 2 | 0.6805 | 0.9612 | 0.5601 |
| 3 | 0.6701 | 0.9610 | 0.5389 |
| 4 | 0.6805 | 0.9609 | 0.5442 |

#### 3. Test Set 最終評估 (Fold 1)

使用驗證集表現最佳的 **Fold 1** 模型在獨立測試集 (5,000 images) 上進行評估：

| 指標 | 數值 | 與 Val 差異 | 評語 |
|:---:|:---:|:---:|:---|
| **mAP** | **0.6960** | -0.0022 | **泛化能力極佳**，幾乎無效能下降 |
| **AUC-Macro** | **0.9634** | +0.0005 | 排序能力甚至略優於驗證集 |
| **F1-Macro** | **0.5803** | -0.0064 | 保持一致 |
| **Precision** | **0.7888** | | 準確率高，誤判率低 |
| **Recall** | **0.5040** | | 召回率仍是主要改進點 |

> **結論**: Test Set 結果 (0.6960) 高度吻合 Validation 結果 (0.6982)，證實模型具有強大的泛化能力，未發生過擬合。

#### 4. 結論與分析
- **穩定性極佳**: AUC 的標準差僅 0.0009，顯示模型在不同資料切分下的排序能力極為一致。
- **Fold 1 表現突出**: mAP 接近 0.70，F1 也最高 (0.58)，值得深入分析其資料分布。
- **改進空間**: F1-Macro 相對較低 (0.55)，顯示可能在某些類別上的 threshold 需要調整，或需要 Per-class 分析來找出弱點。

---


### 實驗記錄 ID: 20260204-ENV-UPGRADE

**日期**: 2026-02-04  
**類別**: 基礎建設與環境設定  
**變更摘要**: 升級開發環境以支援 RTX 5080 (Blackwell)。

#### 1. 變更詳情

- **CUDA 升級**: 從 CUDA 12.4 升級至 CUDA 12.8 Nightly。
- **依賴項更新**: 更新 `torch`、`torchvision` 等核心庫至支援 CUDA 12.8 的 nightly 版本。
- **專案配置**: 修改 `pyproject.toml` 與 `requirements.txt` 以使用最新的 PyTorch nightly index。
- **Git 配置**: 將 `outputs/` 加入 `.gitignore`。

#### 2. 驗證結果

- **硬體相容性**: RTX 5080 運作正常，支援 sm_120 架構。
- **安裝測試**: `uv pip install` 流程順暢。

### 實驗記錄 ID: 20260204-STABILIZE

**日期**: 2026-02-04  
**類別**: 環境穩定性與模型重構  
**變更摘要**: 修正 SigLIP2 載入邏輯與損失函數路徑。

#### 1. 變更詳情

- **SigLIP2 載入**: 採用 `AutoModel` + `AutoImageProcessor` + `GemmaTokenizerFast` 解決 `Siglip2Processor` 的映射 bug。
- **Processor Wrapper**: 新增 `ProcessorWrapper` 類以維持原本的調用介面。
- **損失函數**: 修正 `compute_total_loss` 對 `config` 的存取路徑（去除多餘的 `.loss` 階層）。
- **記憶體資訊**: 優化 `get_gpu_memory_info` 為動態偵測總量。

#### 2. 驗證結果

- **環境檢查**: `scripts/verify_setup.py` 通過所有項目。
- **VRAM 使用**: 凍結 towers 後，Batch size 32 運作正常。

---

### 實驗記錄 ID: 20260206-AB-1-NO-HASH

**日期**: 2026-02-06  
**類別**: Ablation Study (AB-1)  
**變更摘要**: 測試移除 Hash Layer 後的效能表現 (AB-1)。

#### 1. 實驗配置
- **模型**: SigLIP2 + Fusion (Hadamard + Magnitude)
- **Hash Layer**: **❌ 移除 (Skip)**
- **Epochs**: 20
- **Batch Size**: 32 × 2 = 64

#### 2. 實驗結果：No Hash vs Full AGCH

| 指標 | No Hash (AB-1) | Full AGCH | 差異 |
|:---:|:---:|:---:|:---:|
| **mAP** | **0.8323** | 0.6787 | **+0.1536 (+22.6%)** |
| **AUC-Macro** | **0.9834** | 0.9613 | +0.0221 |
| **F1-Macro** | **0.7630** | 0.5522 | +0.2108 |
| **Precision-Macro** | **0.8862** | 0.7728 | +0.1134 |
| **Recall-Macro** | **0.6840** | 0.4755 | +0.2085 |
| **Hamming Loss** | **0.0129** | 0.0190 | -0.0061 |
| **LRAP** | **0.9274** | 0.8476 | +0.0798 |

> [!CRITICAL]
> **Hash Layer 是效能瓶頸的核心原因！**  
> 移除 Hash Layer 後，模型效能 (0.8323) 立即恢復至接近 Baseline MLP (0.8384) 的水準。  
> 這證明 **Direction/Magnitude 分解與 Hadamard Fusion 策略本身是有效的**，問題出在 Hash Layer 的資訊壓縮或損失函數設計上。

#### 3. 結論與行動
1.  **確認主因**: Hash Layer 造成了資訊的嚴重損失或梯度傳遞困難。
2.  **Fusion 有效**: Hadamard + Magnitude Fusion (0.8323) 與 MLP Concat (0.8384) 表現相當，證實此種結構本身無害。
3.  **下一步**: 

---

### 實驗記錄 ID: 20260206-AB-3-BCE-ONLY

**日期**: 2026-02-06  
**類別**: Ablation Study (AB-3)  
**變更摘要**: 保留 Hash Layer 架構但移除所有正則化 Loss (僅使用 BCE)。

#### 1. 實驗配置
- **模型**: SigLIP2 + Fusion + Hash Layer
- **Loss**: **僅 BCE (Binary Cross Entropy)**
    - Cosine Loss Weight: 0.0
    - Hash Loss Weight: 0.0
- **Hash Layer**: ✅ 啟用 (有 Tanh 瓶頸)

#### 2. 實驗結果對比

| 指標 | Baseline (MLP) | AB-1 (No Hash) | AB-3 (BCE Only) | AGCH (Full) |
|:---:|:---:|:---:|:---:|:---:|
| **mAP** | **0.8384** | **0.8323** | 0.7731 | 0.6787 |
| **AUC** | 0.9836 | 0.9834 | 0.9729 | 0.9613 |
| **F1** | 0.7701 | 0.7630 | 0.7030 | 0.5522 |
| **Drop** | - | -0.6% | **-6.5%** | **-15.9%** |

#### 3. 關鍵洞察 (Root Cause Analysis)

透過 AB-1 與 AB-3 的結果，我們可以精確拆解效能損失的來源：

1.  **Hash 瓶頸 (Bottleneck) 造成 ~6.5% 損失**：
    - AB-1 (無 Hash) vs AB-3 (有 Hash 但無 Loss 約束)
    - 僅僅是將特徵通過 Hash Layer (Tanh 壓縮)，mAP 即從 0.83 跌至 0.77。
    - 這表示 64-bit 的資訊容量可能不足，或 Tanh 造成梯度消失。

2.  **Hash/Cosine Loss 造成額外 ~9.4% 損失**：
    - AB-3 (BCE Only) vs AGCH (Full Loss)
    - 加上 `Cosine Loss` 與 `Hash Loss` 後，mAP 從 0.77 暴跌至 0.67。
    - 這證明目前的 **Hash 正則化目標與分類目標存在嚴重衝突 (Competing Objectives)**，為了滿足 Hash 性質（正交、二值化）而犧牲了語義分類能力。

#### 4. 下一步行動建議
*   **立即停止** 當前的 Hash Loss 調優，因為它與主任務衝突。
*   **Pivot 策略**：
    1.  考慮移除 Hash Layer，專注於 AB-1 架構的優化 (Per-class, Ensemble)。
    2.  若必須保留 Hash (為了檢索效率)，需改用 **Product Quantization (PQ)** 或放寬 Hash 約束權重 (e.g., hash_weight 0.1 -> 0.001)。

