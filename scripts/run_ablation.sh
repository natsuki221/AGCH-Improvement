#!/bin/bash
# scripts/run_ablation.sh
# 消融實驗批次執行腳本

set -e

echo "============================================"
echo "🧪 消融實驗批次執行"
echo "============================================"

# 確保在專案根目錄
cd "$(dirname "$0")/.."

# 啟用虛擬環境
source .venv/bin/activate

# AB-1: 無 Hash 層實驗
echo ""
echo ">>> AB-1: 無 Hash 層實驗"
echo "============================================"
python scripts/train.py \
  --config-path ../configs \
  --config-name experiments/cv_experiment \
  model.hash.skip_hash=true \
  training.num_epochs=20 \
  experiment.name=ablation_no_hash

# AB-3: 僅 BCE Loss 實驗
echo ""
echo ">>> AB-3: 僅 BCE Loss 實驗"
echo "============================================"
python scripts/train.py \
  --config-path ../configs \
  --config-name experiments/cv_experiment \
  loss.cosine_weight=0 \
  loss.hash_weight=0 \
  training.num_epochs=20 \
  experiment.name=ablation_bce_only

# AB-4: Hash Bits 變體實驗
echo ""
echo ">>> AB-4: Hash Bits 變體實驗"
echo "============================================"

for bits in 32 64 128 256; do
  echo "  Running hash_bits=$bits..."
  python scripts/train.py \
    --config-path ../configs \
    --config-name experiments/cv_experiment \
    model.hash.bits=$bits \
    training.num_epochs=20 \
    experiment.name=ablation_hash_$bits
done

echo ""
echo "============================================"
echo "✅ 所有消融實驗完成！"
echo "============================================"
echo ""
echo "結果位置:"
echo "  outputs/checkpoints/ablation_no_hash/"
echo "  outputs/checkpoints/ablation_bce_only/"
echo "  outputs/checkpoints/ablation_hash_32/"
echo "  outputs/checkpoints/ablation_hash_64/"
echo "  outputs/checkpoints/ablation_hash_128/"
echo "  outputs/checkpoints/ablation_hash_256/"
