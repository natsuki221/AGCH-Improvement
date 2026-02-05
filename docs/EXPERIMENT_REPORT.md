# 實驗報告

> **專案**: AGCH-Improvement  
> **建立日期**: 2026-02-04  
> **用途**: 記錄所有模型訓練與驗證實驗的結果分析

---

<!-- 新的實驗記錄請加在此處，使用 --- 分隔 -->

### 實驗記錄 ID: 20260205-5FOLD-SIGLIP2

**日期**: 2026-02-05  
**類別**: 5-Fold 交叉驗證 (Cross-Validation)  
**變更摘要**: 完成 SigLIP2 + AGCH 改進版的 5-fold 穩定性與效能驗證。

#### 1. 實驗配置
- **模型**: SigLIP2 (frozen towers) + AGCH
- **Fold 數量**: 5
- **訓練輪數**: 20 Epochs per fold
- **硬體**: RTX 5080 (16GB)

#### 2. 實驗結果 (mAP)
| Fold | 最佳 Epoch | mAP |
|:---:|:---:|:---:|
| Fold 0 | 19 | 0.6644 |
| Fold 1 | 19 | 0.6982 |
| Fold 2 | 17 | 0.6794 |
| Fold 3 | 18 | 0.6701 |
| Fold 4 | 18 | 0.6805 |
| **平均** | - | **0.6785** |
| **標準差** | - | **± 0.012** |

#### 3. 結論與分析
- **穩定性**: 5 個 Fold 均展現一致且穩定的上升趨勢，未發生過擬合或 Mode Collapse。
- **效能**: 平均 mAP 達到 67.85%，在 Fold 1 甚至觸及 69.82%，顯示模型在改進後的 SigLIP2 架構下運作良好。
- **後續建議**: 考量到 Fold 1 與其餘 Fold 的差異，可進一步分析各 Fold 數據分佈或嘗試動態調整學習率。

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
