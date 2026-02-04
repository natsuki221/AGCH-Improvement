#!/bin/bash
# 5-Fold Cross-Validation 自動化執行腳本
# 確保在專案根目錄執行: chmod +x scripts/run_5fold_cv.sh

set -e  # 遇到錯誤立即停止

EXP_NAME="siglip2_cv_run1"
CONFIG_NAME="experiments/cv_experiment"

echo "========================================"
echo "🚀 開始 5-Fold Cross-Validation: $EXP_NAME"
echo "========================================"

# 1. 檢查並生成 Split
if [ ! -f "data/coco/5fold_split.json" ]; then
    echo "生成 Fold 分割檔案..."
    python scripts/create_kfold_split.py
fi

# 2. 記錄開始時間
START_TIME=$(date +%s)

# 3. 依序執行 5 折
for i in {0..4}
do
    echo ""
    echo "─────────────────────────────────────────"
    echo "▶️  正在執行 Fold $i / 4"
    echo "─────────────────────────────────────────"
    
    FOLD_START=$(date +%s)
    
    # 呼叫訓練腳本，覆寫 fold 參數
    python scripts/train.py \
        --config-name $CONFIG_NAME \
        experiments.experiment.name="${EXP_NAME}_fold${i}" \
        experiments.k_fold.current_fold=$i
    
    if [ $? -ne 0 ]; then
        echo "❌ Fold $i 訓練失敗！停止實驗。"
        exit 1
    fi
    
    FOLD_END=$(date +%s)
    FOLD_DURATION=$((FOLD_END - FOLD_START))
    FOLD_HOURS=$((FOLD_DURATION / 3600))
    FOLD_MINS=$(((FOLD_DURATION % 3600) / 60))
    
    echo "✅ Fold $i 完成。耗時: ${FOLD_HOURS}h ${FOLD_MINS}m"
    echo "休息 10 秒釋放 GPU..."
    sleep 10
done

# 4. 計算總時長
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

# 5. 聚合結果
echo ""
echo "─────────────────────────────────────────"
echo "📊 聚合實驗結果..."
echo "─────────────────────────────────────────"
python scripts/aggregate_cv_results.py --exp_prefix "${EXP_NAME}_fold"

echo ""
echo "========================================"
echo "✅ 5-Fold CV 完成！"
echo "總耗時: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo "========================================"