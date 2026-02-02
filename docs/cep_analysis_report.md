# CEP 專案改進分析報告

**分析日期**: 2026-01-27  
**專案**: CEP (Comparison-Experiment-Project) CLI Tool  
**版本比較**: 原始設計 (cep.zip) vs 修改規劃 (cep-fix.zip)

---

## 執行摘要 (Executive Summary)

### 核心變更方向
修改版 (cep-fix) 將專案定位從「**實驗執行工具**」重新聚焦為「**資料匯入與分析工具**」。這個轉變反映了實際開發需求的演化：從「如何跑實驗」轉向「如何分析已有的實驗結果」。

### 整體評價
**✅ 大部分改進合理且必要**，但存在一些潛在風險與可優化空間。

---

## 一、架構設計改進分析

### 1.1 資料庫設計 ✅ **優秀改進**

#### 原始設計問題
- 僅提及使用 SQLite，但未定義具體 Schema
- 缺乏對「如何儲存實驗結果」的明確規範
- 沒有處理資料持久化的策略

#### 修改版改進
```sql
-- 新增 Zero-ETL 策略：原始資料表
CREATE TABLE experiment_match (
    experiment_id TEXT NOT NULL,
    patent_number TEXT NOT NULL,
    comparison_type TEXT NOT NULL,
    project_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    rank INTEGER,
    UNIQUE(experiment_id, patent_number, project_id, comparison_type)
);

-- 新增聚合統計表
CREATE TABLE patent_stats (
    experiment_id TEXT NOT NULL,
    patent_number TEXT NOT NULL,
    top1_score REAL NOT NULL,
    top2_score REAL,
    margin REAL,
    match_count INTEGER NOT NULL,
    risk_score REAL
);
```

**優點**:
1. **Zero-ETL 策略**: 直接映射 JSON 結構，減少轉換錯誤
2. **雙表設計**: 分離原始資料與聚合統計，提升查詢效能
3. **冪等性支援**: UNIQUE 複合鍵確保重複匯入安全
4. **WAL Mode**: 支援並發讀取，適合分析場景

**潛在問題**:
⚠️ **缺少資料版本控制**: 如果實驗配置改變（如模型版本、參數），舊資料可能與新資料混淆

**建議改進**:
```sql
-- 建議在 experiment_match 加入版本欄位
ALTER TABLE experiment_match ADD COLUMN schema_version TEXT DEFAULT 'v1';
ALTER TABLE experiment_match ADD COLUMN model_version TEXT; -- 語意模型版本
ALTER TABLE experiment_match ADD COLUMN metadata JSON; -- 實驗配置快照
```

### 1.2 統計指標設計 ✅ **創新且實用**

#### 新增核心指標: Margin (信心差距)

**定義**: `Margin = Top1_Score - Top2_Score`

**創新價值**:
- 解決了「分數 0.85 vs 0.82 有什麼實質差異」的問題
- 提供「模型確定性」的量化指標
- 符合研發處「需要明確證據」的行政需求

**實際應用場景**:
```
專利 A: Top1=0.90, Top2=0.35 → Margin=0.55 (高信心)
專利 B: Top1=0.90, Top2=0.88 → Margin=0.02 (低信心)
```
→ 雖然兩者分數都高，但專利 A 的風險更明確

**邊界處理** ✅:
修改版正確處理了以下情況：
- 只有 1 筆匹配 → Top2=0，Margin=Top1
- 負分數（Mean-Centered）→ 正常計算
- 零筆匹配 → 跳過並記錄警告

---

## 二、功能範圍變更分析

### 2.1 核心功能對比

| 功能類別 | 原始設計 (cep.zip) | 修改版 (cep-fix.zip) | 評價 |
|---------|-------------------|---------------------|------|
| **實驗執行** | ✅ 主要功能 (`run-experiment`) | ⚠️ 延後至 Phase 2 | 合理聚焦 |
| **資料匯入** | ❌ 未提及 | ✅ 核心功能 (`ingest`) | **關鍵補充** |
| **統計分析** | ⚠️ 僅提及產出 JSON | ✅ 深度指標（Margin, P90） | **大幅增強** |
| **報告生成** | ⚠️ JSON/Markdown 基礎 | ✅ 多格式（MD/JSON/CSV） | 改進 |
| **隨機基準線** | ✅ 明確要求 | ❌ **缺失** | **重大遺漏** ⚠️ |

### 2.2 關鍵功能缺失 ⚠️ **需補強**

#### 問題：隨機基準線 (Random Baseline) 功能遺失

**原始設計 (PRD Section 3.1)**:
```markdown
**無標註證明力 (Zero-Shot Validation)**
透過 Random Baseline Lift 證明系統具備統計顯著的辨識力。
- 指標: P50(Risk_Real) - P95(Risk_Random) > threshold
```

**修改版狀態**: 
❌ 在 PRD、Architecture、Epics 中**完全未提及**

**影響評估**:
1. **技術驗證缺失**: 無法證明模型優於隨機猜測
2. **業務價值損失**: 研發處無法在「缺乏標註」情況下驗證系統
3. **創新點削弱**: 原 PRD 強調的「零樣本統計驗證」特色消失

**建議修復**:
```python
# 需在 core/stats.py 補充
class RandomBaselineCalculator:
    """計算隨機負採樣的基準線分佈"""
    
    def generate_random_pairs(
        self, 
        patent_count: int, 
        project_count: int, 
        sample_size: int = 1000
    ) -> List[Tuple[str, str]]:
        """隨機產生專利-計畫配對"""
        pass
    
    def calculate_lift(
        self, 
        real_scores: List[float], 
        random_scores: List[float]
    ) -> Dict[str, float]:
        """計算提升度 (Lift)"""
        return {
            "real_p50": np.percentile(real_scores, 50),
            "random_p95": np.percentile(random_scores, 95),
            "lift": np.percentile(real_scores, 50) - np.percentile(random_scores, 95)
        }
```

**補充建議 Epic**:
```markdown
### Epic 4: 統計驗證與基準線分析 (Validation & Baseline)
**目標**: 實作隨機基準線計算，提供零樣本驗證能力

#### Story 4.1: 隨機負採樣引擎
- 產生 N 組隨機專利-計畫配對
- 計算這些配對的相似度分佈
- 輸出 P50, P90, P95 統計值

#### Story 4.2: Lift 指標計算與報告
- 比對真實實驗與隨機基準的分佈
- 計算統計顯著性 (Lift)
- 在 Compare 報告中加入 Baseline 比較表格
```

---

## 三、技術架構改進評估

### 3.1 CLI 指令設計 ✅ **清晰改進**

#### 原始設計
```bash
# 模糊的指令結構
python scripts/run_comparison.py
  - run-experiment
  - validate
  - generate-report
  - config
```

#### 修改版
```bash
cep ingest --input-file exp-1.json --type semantic
cep compare exp-1 exp-2 exp-3 --output report.md
cep stats --exp-id exp-1 --metric margin
```

**改進點**:
1. ✅ 語意更清晰（ingest vs run-experiment）
2. ✅ 參數命名一致（kebab-case）
3. ✅ 職責分離明確

### 3.2 程式碼組織 ✅ **結構優化**

#### 分層架構改進
```
原始 (混亂):
cep-cli/
├── main.py (混合邏輯)
├── experiment.py
└── report.py

修改版 (清晰):
cep-cli/
├── commands/     # CLI 介面層 (UI)
│   ├── ingest.py
│   ├── compare.py
├── core/         # 領域邏輯層 (Pure Python)
│   ├── stats.py
│   ├── pipeline.py
└── models/       # 資料層 (SQLModel)
    ├── raw.py
    └── stats.py
```

**優點**:
- 關注點分離 (Separation of Concerns)
- 可測試性提升（core/ 不依賴框架）
- 符合 Clean Architecture 原則

### 3.3 錯誤處理策略 ✅ **專業化**

**新增自定義異常體系**:
```python
# core/exceptions.py
class IngestError(Exception): pass
class ConfigError(Exception): pass
class ValidationError(Exception): pass
```

**分層處理模式**:
```
core/     → 拋出自定義異常
commands/ → 捕捉並轉換為 Rich 格式輸出
          → 回傳正確的 Exit Code
```

---

## 四、非功能性需求評估

### 4.1 效能指標 ⚠️ **過於樂觀**

#### 宣稱目標
```
- Ingest 速度: > 5000 records/sec
- 記憶體峰值: < 500MB (處理 100MB JSON)
- CLI 回應: < 0.5 秒
```

#### 潛在問題
1. **5000 records/sec 不切實際**:
   - SQLite 單執行緒寫入限制：約 500-1000 inserts/sec
   - 即使使用 Batch Insert（1000 筆/transaction），實際速度約 2000-3000/sec

2. **建議修正**:
```markdown
### 修正後的效能指標
- **Ingest 速度**: > 2000 records/sec (Batch Insert, 1000 rows/txn)
- **記憶體**: < 500MB (使用 ijson 串流解析)
- **CLI Startup**: < 0.5 秒 (合理)
```

### 4.2 測試策略 ✅ **完善**

#### Snapshot Testing 創新應用
```python
# tests/snapshots/test_stats/
exp-1-margin.json  # 基準快照
```

**優點**:
- 解決「缺乏 Golden Dataset」問題
- 確保重構後數值一致性
- 適合科學計算場景

**建議加強**:
```python
# 補充回歸測試
def test_margin_backward_compatibility():
    """確保新 CLI 與舊腳本輸出數值一致"""
    legacy_result = load_json("legacy_exp-1-result.json")
    new_result = run_cep_compare("exp-1")
    
    assert_snapshots_match(
        legacy_result["patent_level_analysis"]["margin_p90"],
        new_result["margin_p90"],
        tolerance=1e-6
    )
```

---

## 五、專案管理與風險

### 5.1 MVP 範圍界定 ✅ **務實**

#### 原始設計 (過於激進)
```
MVP = 實驗執行 + 隨機基準線 + 報告 + JSON 升級
```

#### 修改版 (聚焦核心)
```
Phase 1 (MVP):  資料匯入 + 統計分析 + 比對報告
Phase 2:        實驗執行整合
Phase 3:        互動式報告 + 進階分析
```

**評價**: ✅ 符合「Replacement MVP」策略，降低初期風險

### 5.2 風險緩解 ⚠️ **需補充**

#### 已識別風險
1. ✅ SQLite 效能 → 緩解：Batch Insert + WAL Mode
2. ✅ 使用者信任 → 緩解：Snapshot Testing 驗證
3. ✅ Schema 對齊 → 緩解：Zero-ETL 策略

#### 缺失風險
⚠️ **資料遷移風險**:
- 問題：如何處理「舊實驗結果格式」與「新 DB Schema」不匹配？
- 建議：
```python
# 建議加入遷移工具
cep migrate --from-version v1 --to-version v2 --dry-run
```

⚠️ **並發寫入風險**:
- 問題：Single-Writer Policy 在多人協作場景下的限制
- 建議：
```python
# 加入鎖定機制
class SQLiteWriteLock:
    def __enter__(self):
        # 檢查 .lock 檔案
        if lock_exists():
            raise RuntimeError("Another ingest process is running")
        create_lock()
    
    def __exit__(self):
        remove_lock()
```

---

## 六、具體改進建議

### 6.1 高優先級 (Critical)

#### 🔴 補充隨機基準線功能
```markdown
**位置**: Epic 4 (新增)
**工作量**: 3-5 個 Story
**價值**: 恢復原 PRD 的核心創新點
```

**實作要點**:
1. 在 `core/stats.py` 新增 `RandomSampler` 類別
2. 在 `compare` 指令加入 `--include-baseline` 參數
3. 報告中加入 Lift 指標與統計顯著性檢驗

#### 🔴 修正效能預期
```diff
- Ingest 速度: > 5000 records/sec
+ Ingest 速度: > 2000 records/sec (實測驗證)
```

### 6.2 中優先級 (Important)

#### 🟡 資料版本控制
```sql
-- 在 experiment_match 加入
ALTER TABLE experiment_match ADD COLUMN 
    metadata JSON DEFAULT '{}';
```

**儲存內容**:
```json
{
  "model_version": "jina-v4",
  "threshold": 0.85,
  "embedding_dim": 768,
  "timestamp": "2026-01-27T10:30:00Z"
}
```

#### 🟡 並發控制機制
```python
# config.py
INGEST_LOCK_TIMEOUT = 300  # 5 分鐘
INGEST_LOCK_FILE = ".cep_ingest.lock"
```

### 6.3 低優先級 (Nice to Have)

#### 🟢 互動式查詢介面
```bash
cep query "SELECT * FROM patent_stats WHERE margin > 0.3"
```

#### 🟢 自動化驗證報告
```bash
cep validate exp-1 --against-snapshot baseline.json
```

---

## 七、資料庫最佳化建議

### 7.1 索引策略優化

#### 當前設計
```sql
CREATE INDEX idx_exp_match_exp_id ON experiment_match(experiment_id);
CREATE INDEX idx_exp_match_patent ON experiment_match(patent_number);
```

#### 建議補充
```sql
-- 複合索引優化常見查詢
CREATE INDEX idx_exp_patent_type ON experiment_match(
    experiment_id, patent_number, comparison_type
);

-- 覆蓋索引 (Covering Index) 加速聚合
CREATE INDEX idx_score_lookup ON experiment_match(
    experiment_id, patent_number, similarity_score DESC
) WHERE comparison_type = 'semantic';

-- 部分索引 (Partial Index) 減少空間
CREATE INDEX idx_high_risk ON patent_stats(risk_score DESC)
WHERE risk_score > 0.5;
```

### 7.2 查詢效能優化

#### 當前的 Margin 計算邏輯問題
```python
# 可能的低效實作 (N+1 Query)
for patent in patents:
    scores = db.query(
        "SELECT similarity_score FROM experiment_match 
         WHERE patent_number = ?", patent
    ).all()
    margin = scores[0] - scores[1]  # 多次查詢
```

#### 建議改用單一 SQL
```sql
-- 使用 Window Function (需 SQLite 3.25+)
WITH ranked_scores AS (
    SELECT 
        patent_number,
        similarity_score,
        ROW_NUMBER() OVER (
            PARTITION BY experiment_id, patent_number 
            ORDER BY similarity_score DESC
        ) as rank
    FROM experiment_match
    WHERE experiment_id = ?
)
SELECT 
    patent_number,
    MAX(CASE WHEN rank = 1 THEN similarity_score END) as top1,
    MAX(CASE WHEN rank = 2 THEN similarity_score END) as top2,
    MAX(CASE WHEN rank = 1 THEN similarity_score END) - 
    COALESCE(MAX(CASE WHEN rank = 2 THEN similarity_score END), 0) as margin
FROM ranked_scores
GROUP BY patent_number;
```

**效能提升**: 從 O(N) 次查詢降至 O(1) 次

---

## 八、文件品質評估

### 8.1 PRD 文件

| 評估項目 | 原始版本 | 修改版本 | 評分 |
|---------|---------|---------|------|
| **需求完整性** | 80% | 75% | ⚠️ 下降 (缺失 Random Baseline) |
| **用戶旅程** | 清晰 | 非常清晰 | ✅ 改進 |
| **技術規格** | 模糊 | 詳細 | ✅ 大幅改進 |
| **驗收標準** | 基礎 | 量化明確 | ✅ 改進 |

### 8.2 Architecture 文件

| 評估項目 | 評分 | 說明 |
|---------|------|------|
| **決策記錄** | ⭐⭐⭐⭐⭐ | AD-01~AD-08 完整且有理由 |
| **實作指引** | ⭐⭐⭐⭐⭐ | 目錄結構、命名規範清晰 |
| **邊界定義** | ⭐⭐⭐⭐ | 邊界條件表格完善 |
| **可執行性** | ⭐⭐⭐⭐ | 缺少部分 SQL 範例 |

### 8.3 Epics 文件

**優點**:
- ✅ Story 顆粒度適中（單次 Session 可完成）
- ✅ 驗收準則 (AC) 明確可驗證
- ✅ 需求追溯矩陣 (FR Coverage Matrix) 完整

**建議改進**:
```markdown
## Epic 4: 統計驗證與基準線分析 (新增)

### Story 4.1: 隨機負採樣引擎
**User Story**: 身為研究員，我想要看到模型與隨機猜測的比較，以證明系統價值。

**AC**:
1. 實作 `core/baseline.py` 的隨機配對生成
2. 計算 P50(Real) - P95(Random) 的 Lift 指標
3. 若 Lift < 0.1，在報告中標註警告
```

---

## 九、總結與建議

### 9.1 整體評價

**✅ 強項**:
1. 資料庫設計 (Zero-ETL + 雙表策略) 極佳
2. CLI 介面設計清晰且符合 Unix 哲學
3. 程式碼組織符合現代最佳實踐
4. Margin 指標創新且實用

**⚠️ 需改進**:
1. 缺失隨機基準線功能（原 PRD 核心特色）
2. 效能指標過於樂觀（5000 records/sec 不切實際）
3. 缺乏資料版本控制機制
4. 並發寫入保護不足

**❌ 潛在風險**:
1. 功能範圍變更過大（從「實驗執行」到「資料分析」）
2. 未明確說明與原系統的關係（是取代還是補充？）

### 9.2 行動建議 (Action Items)

#### 立即執行 (Before Sprint 1)
1. ✅ **補充 Epic 4**: 隨機基準線驗證
2. ✅ **修正效能指標**: 從 5000 降至 2000 records/sec
3. ✅ **加入版本控制欄位**: `metadata JSON` in `experiment_match`

#### Sprint 1 期間
4. ✅ 實作並測試 Batch Insert 實際速度
5. ✅ 建立 Snapshot Testing 基準資料集
6. ✅ 撰寫資料遷移腳本 (`cep migrate`)

#### Sprint 2 前評估
7. ⚠️ 決定是否保留「實驗執行」功能（或永久分離為獨立工具）
8. ⚠️ 確認與原 `compare_experiments.py` 的整合/取代策略

### 9.3 最終建議

**建議採用修改版架構**，但需補充以下內容：

```markdown
### 必要補充清單
1. Epic 4: 隨機基準線分析 (恢復原 PRD 特色)
2. Story X.X: 資料版本控制與 Metadata 記錄
3. Story X.X: 並發控制與鎖定機制
4. Story X.X: 效能基準測試與驗證

### 建議修改
- 降低 Ingest 速度目標至 2000 records/sec
- 明確 Phase 2 的實驗執行整合計畫
- 補充資料遷移策略與向後相容性說明
```

---

## 附錄：關鍵技術決策對照表

| 決策點 | 原始設計 | 修改版 | 評價 | 建議 |
|-------|---------|--------|------|------|
| **主要目標** | 實驗執行工具 | 資料分析工具 | ✅ 聚焦合理 | 維持 |
| **資料庫 Schema** | 未定義 | Zero-ETL 雙表 | ✅ 優秀 | 加入 metadata |
| **CLI 架構** | 基礎 Typer | 分層結構 | ✅ 改進 | 維持 |
| **隨機基準線** | ✅ 必要 | ❌ 缺失 | ⚠️ 重大遺漏 | **必須補充** |
| **Margin 指標** | 未提及 | ✅ 核心創新 | ✅ 優秀 | 維持 |
| **效能目標** | 未定義 | 5000/sec | ⚠️ 過高 | 降至 2000/sec |
| **測試策略** | 基礎 | Snapshot Testing | ✅ 創新 | 維持 |
| **並發控制** | 未提及 | Single-Writer | ⚠️ 不足 | 加入 Lock 機制 |

---

**報告結論**: 修改版在架構設計與工程實踐上有顯著提升，但需補充隨機基準線功能、修正效能預期、加強資料版本控制，才能完整達成原始 PRD 的業務目標。

**信心等級**: ⭐⭐⭐⭐ (4/5)  
**建議執行**: ✅ 採用，但需修正上述問題
