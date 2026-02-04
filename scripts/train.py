# scripts/train.py
"""
訓練腳本 - AGCH-Improvement
支援 RTX 5080 16GB，使用混合精度訓練與梯度累積
"""

import os
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    hamming_loss,
    coverage_error,
    label_ranking_loss,
    label_ranking_average_precision_score,
)

# 本專案模組
from siglip2_multimodal_hash.model import MultimodalHashKNN
from siglip2_multimodal_hash.dataset import create_dataloader
from siglip2_multimodal_hash.losses import compute_total_loss
from siglip2_multimodal_hash.utils import set_seed, get_gpu_memory_info


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config):
    """訓練一個 epoch"""
    model.train()

    total_losses = {"total": 0, "bce": 0, "cos": 0, "hash": 0}
    num_batches = 0
    accumulation_steps = config.training.gradient_accumulation_steps

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        # 移動資料到 GPU
        pixel_values = batch["pixel_values"].cuda()
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        labels = batch["labels"].cuda()

        # 混合精度前向傳播
        with autocast(enabled=config.memory_optimization.mixed_precision):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_components=True,
            )

            loss_dict = compute_total_loss(outputs=outputs, labels=labels, config=config.loss)

            loss = loss_dict["total"] / accumulation_steps

        # 反向傳播
        scaler.scale(loss).backward()

        # 梯度累積
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # 累計損失
        total_losses["total"] += loss_dict["total"].item()
        total_losses["bce"] += loss_dict["bce"].item()
        total_losses["cos"] += loss_dict["cos"].item()
        total_losses["hash"] += loss_dict["hash"].item()
        num_batches += 1

        # 更新進度條
        pbar.set_postfix(
            {
                "loss": f"{loss_dict['total'].item():.4f}",
                "GPU": f"{get_gpu_memory_info()['allocated_gb']:.1f}GB",
            }
        )

    # 平均損失
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


@torch.no_grad()
def validate(model, val_loader, config):
    """驗證模型，計算多標籤分類常用指標"""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation"):
        # 移動資料到 GPU
        pixel_values = batch["pixel_values"].cuda()
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        labels = batch["labels"].cuda()

        # 前向傳播
        with autocast(enabled=config.memory_optimization.mixed_precision):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_components=True,
            )

            loss_dict = compute_total_loss(outputs=outputs, labels=labels, config=config.loss)

        total_loss += loss_dict["total"].item()
        num_batches += 1

        # 收集預測
        probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    # 合併結果
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pred_binary = (all_preds > 0.5).astype(int)

    # ============================================================
    # 多標籤分類研究常用指標
    # ============================================================

    # 1. Mean Average Precision (mAP) - 主要指標
    mAP = average_precision_score(all_labels, all_preds, average="macro")

    # 2. AUC-ROC (macro/micro)
    try:
        auc_macro = roc_auc_score(all_labels, all_preds, average="macro")
        auc_micro = roc_auc_score(all_labels, all_preds, average="micro")
    except ValueError:
        # 某些標籤可能沒有正例
        auc_macro = 0.0
        auc_micro = 0.0

    # 3. F1 Scores
    f1_micro = f1_score(all_labels, pred_binary, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, pred_binary, average="macro", zero_division=0)

    # 4. Precision & Recall
    precision_micro = precision_score(all_labels, pred_binary, average="micro", zero_division=0)
    precision_macro = precision_score(all_labels, pred_binary, average="macro", zero_division=0)
    recall_micro = recall_score(all_labels, pred_binary, average="micro", zero_division=0)
    recall_macro = recall_score(all_labels, pred_binary, average="macro", zero_division=0)

    # 5. Hamming Loss (越低越好)
    h_loss = hamming_loss(all_labels, pred_binary)

    # 6. Ranking 指標
    try:
        # Coverage Error: 平均需要包含多少標籤才能涵蓋所有真實標籤
        cov_error = coverage_error(all_labels, all_preds)
        # Label Ranking Loss: 排序損失 (越低越好)
        rank_loss = label_ranking_loss(all_labels, all_preds)
        # Label Ranking Average Precision (LRAP)
        lrap = label_ranking_average_precision_score(all_labels, all_preds)
    except ValueError:
        cov_error = 0.0
        rank_loss = 0.0
        lrap = 0.0

    # 7. Mean Absolute Error (預測機率與真實標籤差距)
    mae = np.mean(np.abs(all_preds - all_labels))

    return {
        # 主要指標
        "loss": total_loss / num_batches,
        "mAP": mAP,
        # AUC
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        # F1
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        # Precision & Recall
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        # 其他
        "hamming_loss": h_loss,
        "coverage_error": cov_error,
        "ranking_loss": rank_loss,
        "lrap": lrap,
        "mae": mae,
    }


@hydra.main(config_path="../configs", config_name="hardware/rtx5080_16gb", version_base=None)
def main(raw_config: DictConfig):
    """主訓練函數"""
    # Hydra 會將配置包在資料夾名下，需要解開
    if "hardware" in raw_config:
        config = raw_config.hardware
    elif "experiments" in raw_config:
        config = raw_config.experiments
    else:
        config = raw_config

    print("=" * 60)
    print("AGCH-Improvement 訓練腳本")
    print("=" * 60)

    # 顯示配置
    print(f"\n📋 實驗: {config.experiment.name}")
    print(
        f"📋 Batch size: {config.training.batch_size} x {config.training.gradient_accumulation_steps} = {config.training.batch_size * config.training.gradient_accumulation_steps}"
    )
    print(f"📋 Epochs: {config.training.num_epochs}")

    # 設定 seed
    set_seed(config.experiment.seed)

    # 檢查 CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用！此腳本需要 GPU")

    print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"🖥️  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 初始化 wandb (可選)
    if config.logging.use_wandb:
        try:
            import wandb

            wandb.init(
                project=config.logging.wandb_project,
                entity=(
                    config.logging.wandb_entity
                    if config.logging.wandb_entity != "your-username"
                    else None
                ),
                config=OmegaConf.to_container(config, resolve=True),
                name=config.experiment.name,
            )
            use_wandb = True
        except Exception as e:
            print(f"⚠️  Wandb 初始化失敗: {e}")
            use_wandb = False
    else:
        use_wandb = False

    # 建立模型
    print("\n📦 建立模型...")
    model = MultimodalHashKNN(config.model).cuda()

    # 顯示記憶體資訊
    mem_info = get_gpu_memory_info()
    print(
        f"✓ 模型載入後 GPU 記憶體: {mem_info['allocated_gb']:.2f}GB / {mem_info['total_gb']:.1f}GB"
    )

    # 計算可訓練參數
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"✓ 可訓練參數: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)"
    )

    # 建立 DataLoader
    print("\n📂 建立 DataLoader...")

    # 檢查是否為 K-Fold 模式
    use_k_fold = config.get("k_fold", {}).get("enabled", False)
    fold_idx = config.get("k_fold", {}).get("current_fold", None) if use_k_fold else None

    if use_k_fold:
        print(f"📋 K-Fold 模式: Fold {fold_idx}")

    train_loader = create_dataloader(config, split="train", fold_idx=fold_idx)
    val_loader = create_dataloader(config, split="val", fold_idx=fold_idx)
    print(f"✓ 訓練集: {len(train_loader)} batches")
    print(f"✓ 驗證集: {len(val_loader)} batches")

    # 建立 optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
    )

    # 建立 scheduler
    total_steps = (
        len(train_loader)
        * config.training.num_epochs
        // config.training.gradient_accumulation_steps
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.scheduler.min_lr
    )

    # 混合精度 scaler
    scaler = GradScaler(enabled=config.memory_optimization.mixed_precision)

    # 建立儲存目錄 - K-Fold 模式使用實驗名稱作為子目錄
    base_save_dir = Path("outputs/checkpoints")
    exp_name = config.experiment.name
    save_dir = base_save_dir / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 訓練迴圈
    best_val_map = 0
    patience_counter = 0

    print("\n🚀 開始訓練...")
    for epoch in range(config.training.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_losses = train_epoch(model, train_loader, optimizer, scheduler, scaler, config)
        print(
            f"Train Loss: {train_losses['total']:.4f} "
            f"(BCE: {train_losses['bce']:.4f}, "
            f"Cos: {train_losses['cos']:.4f}, "
            f"Hash: {train_losses['hash']:.4f})"
        )

        # 驗證
        val_metrics = validate(model, val_loader, config)
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"mAP: {val_metrics['mAP']:.4f}, "
            f"AUC: {val_metrics['auc_macro']:.4f}, "
            f"F1-Macro: {val_metrics['f1_macro']:.4f}"
        )
        print(
            f"     Precision: {val_metrics['precision_macro']:.4f}, "
            f"Recall: {val_metrics['recall_macro']:.4f}, "
            f"Hamming: {val_metrics['hamming_loss']:.4f}, "
            f"MAE: {val_metrics['mae']:.4f}"
        )

        # 記錄到 wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_losses["total"],
                    "train/loss_bce": train_losses["bce"],
                    "train/loss_cos": train_losses["cos"],
                    "train/loss_hash": train_losses["hash"],
                    "val/loss": val_metrics["loss"],
                    "val/mAP": val_metrics["mAP"],
                    "val/auc_macro": val_metrics["auc_macro"],
                    "val/auc_micro": val_metrics["auc_micro"],
                    "val/f1_micro": val_metrics["f1_micro"],
                    "val/f1_macro": val_metrics["f1_macro"],
                    "val/precision_macro": val_metrics["precision_macro"],
                    "val/recall_macro": val_metrics["recall_macro"],
                    "val/hamming_loss": val_metrics["hamming_loss"],
                    "val/ranking_loss": val_metrics["ranking_loss"],
                    "val/lrap": val_metrics["lrap"],
                    "val/mae": val_metrics["mae"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "gpu_memory_gb": get_gpu_memory_info()["allocated_gb"],
                }
            )

        # 儲存最佳模型
        if val_metrics["mAP"] > best_val_map:
            best_val_map = val_metrics["mAP"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,  # 所有評估指標
                "val_mAP": val_metrics["mAP"],  # 向後兼容
                "config": OmegaConf.to_container(config, resolve=True),
            }
            checkpoint_path = save_dir / "best_model.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ 儲存最佳模型: {checkpoint_path} (mAP: {val_metrics['mAP']:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print(f"最佳 Val mAP: {best_val_map:.4f}")
    print("=" * 60)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
