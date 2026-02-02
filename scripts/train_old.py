# scripts/train.py

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import sys

# 加入 src 到 Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from siglip2_multimodal_hash.model import MultimodalHashKNN
from siglip2_multimodal_hash.dataset import create_dataloader
from siglip2_multimodal_hash.losses import compute_total_loss
from siglip2_multimodal_hash.utils import get_gpu_memory_info, MemoryMonitor, set_seed


def train_epoch(
    model: MultimodalHashKNN,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    config: DictConfig,
    epoch: int,
) -> dict:
    """訓練一個 epoch"""

    model.train()
    total_losses = {"total": 0, "bce": 0, "cos": 0, "hash": 0}

    accumulation_steps = config.training.gradient_accumulation_steps
    memory_monitor = MemoryMonitor(config.memory_optimization.alert_vram_threshold_gb)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch_idx, batch in enumerate(pbar):
        # 移到 GPU
        pixel_values = batch["pixel_values"].to("cuda", non_blocking=True)
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
        labels = batch["labels"].to("cuda", non_blocking=True)

        # ⚠️ 混合精度前向傳播
        with autocast(dtype=torch.float16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_components=True,
            )

            # 計算損失
            loss, loss_dict = compute_total_loss(outputs, labels, config)
            loss = loss / accumulation_steps  # 梯度累積

        # 反向傳播
        scaler.scale(loss).backward()

        # ⚠️ 梯度累積
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

            # 更新參數
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 累積損失
        for key in total_losses:
            total_losses[key] += loss_dict[key]

        # 更新進度條
        pbar.set_postfix(
            {
                "loss": loss_dict["total"],
                "bce": loss_dict["bce"],
                "mem": f"{memory_monitor.get_stats()['gpu']['allocated_gb']:.1f}GB",
            }
        )

        # ⚠️ 定期監控記憶體
        if batch_idx % config.logging.log_every_n_steps == 0:
            if config.memory_optimization.log_gpu_memory:
                memory_monitor.print_stats(f"Batch {batch_idx} - ")

        # ⚠️ 定期清理快取
        if batch_idx % config.memory_optimization.empty_cache_steps == 0:
            torch.cuda.empty_cache()

    # 更新學習率
    scheduler.step()

    # 返回平均損失
    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(model: MultimodalHashKNN, dataloader, config: DictConfig) -> dict:
    """驗證"""

    model.eval()
    total_losses = {"total": 0, "bce": 0, "cos": 0, "hash": 0}
    all_logits = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Validating")

    for batch in pbar:
        pixel_values = batch["pixel_values"].to("cuda", non_blocking=True)
        input_ids = batch["input_ids"].to("cuda", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
        labels = batch["labels"].to("cuda", non_blocking=True)

        with autocast(dtype=torch.float16):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_components=True,
            )

            loss, loss_dict = compute_total_loss(outputs, labels, config)

        # 累積
        for key in total_losses:
            total_losses[key] += loss_dict[key]

        all_logits.append(outputs["logits"].cpu())
        all_labels.append(labels.cpu())

    # 計算指標
    from sklearn.metrics import average_precision_score, f1_score

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    y_true = all_labels.numpy()
    y_scores = torch.sigmoid(all_logits).numpy()
    y_pred = (y_scores > 0.5).astype(int)

    metrics = {
        "loss": total_losses["total"] / len(dataloader),
        "mAP": average_precision_score(y_true, y_scores, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    return metrics


@hydra.main(version_base=None, config_path="../configs/hardware", config_name="rtx5080_16gb")
def main(config: DictConfig):
    """主訓練函數"""

    # 顯示配置
    print("=" * 60)
    print("訓練配置")
    print("=" * 60)
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # 設定 seed
    set_seed(config.experiment.seed)

    # 初始化 wandb
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=config.experiment.name,
        )

    # 建立模型
    print("\n建立模型...")
    model = MultimodalHashKNN(config).cuda()

    # 顯示記憶體資訊
    mem_info = get_gpu_memory_info()
    print(f"模型載入後 GPU 記憶體: {mem_info['allocated_gb']:.2f}GB / 16GB")

    # 建立 DataLoader
    print("\n建立 DataLoader...")
    train_loader = create_dataloader(config, split="train")
    val_loader = create_dataloader(config, split="val")

    # 建立 optimizer 與 scheduler
    print("\n建立 optimizer 與 scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )

    from transformers import get_cosine_schedule_with_warmup

    num_training_steps = len(train_loader) * config.training.num_epochs
    num_warmup_steps = int(num_training_steps * config.scheduler.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=1,
    )

    # 混合精度 scaler
    scaler = GradScaler()

    # 訓練迴圈
    print("\n開始訓練...")
    print("=" * 60)

    best_val_map = 0
    patience_counter = 0

    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        print("-" * 60)

        # 訓練
        train_losses = train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch)

        print(
            f"Train - Loss: {train_losses['total']:.4f} "
            f"(BCE: {train_losses['bce']:.4f}, "
            f"Cos: {train_losses['cos']:.4f}, "
            f"Hash: {train_losses['hash']:.4f})"
        )

        # 驗證
        if (epoch + 1) % config.training.val_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, config)

            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"mAP: {val_metrics['mAP']:.4f}, "
                f"F1-Micro: {val_metrics['f1_micro']:.4f}, "
                f"F1-Macro: {val_metrics['f1_macro']:.4f}"
            )

            # 記錄到 wandb
            if config.logging.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_losses["total"],
                        "train/loss_bce": train_losses["bce"],
                        "train/loss_cos": train_losses["cos"],
                        "train/loss_hash": train_losses["hash"],
                        "val/loss": val_metrics["loss"],
                        "val/mAP": val_metrics["mAP"],
                        "val/f1_micro": val_metrics["f1_micro"],
                        "val/f1_macro": val_metrics["f1_macro"],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # 儲存最佳模型
            if val_metrics["mAP"] > best_val_map:
                best_val_map = val_metrics["mAP"]
                patience_counter = 0

                # 儲存 checkpoint
                checkpoint_dir = Path(config.checkpointing.save_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = (
                    checkpoint_dir / f"best_model_epoch{epoch+1}_mAP{val_metrics['mAP']:.4f}.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mAP": val_metrics["mAP"],
                        "config": OmegaConf.to_container(config, resolve=True),
                    },
                    checkpoint_path,
                )

                print(f"✓ 儲存最佳模型: {checkpoint_path}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print("\n" + "=" * 60)
    print("訓練完成！")
    print(f"最佳 Val mAP: {best_val_map:.4f}")
    print("=" * 60)

    if config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
