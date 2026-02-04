# 實驗報告

> **專案**: AGCH-Improvement  
> **建立日期**: 2026-02-04  
> **用途**: 記錄所有模型訓練與驗證實驗的結果分析

---

<!-- 新的實驗記錄請加在此處，使用 --- 分隔 -->

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
