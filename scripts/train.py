# scripts/train.py
import torch
import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm


@hydra.main(config_path="../configs", config_name="hardware/rtx5080_16gb")
def main(config: DictConfig):
    # 設定 seed
    torch.manual_seed(config.experiment.seed)

    # 初始化 wandb
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            config=dict(config),
            name=config.experiment.name,
        )

    # 建立模型
    print("建立模型...")
    model = MultimodalHashKNN(config.model).cuda()

    # 顯示記憶體資訊
    mem_info = get_gpu_memory_info()
    print(f"模型載入後 GPU 記憶體: {mem_info['allocated_gb']:.2f}GB / 16GB")

    # 建立 DataLoader
    print("建立 DataLoader...")
    train_loader = create_dataloader(config, split="train")
    val_loader = create_dataloader(config, split="val")

    # 建立 optimizer 與 scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        betas=config.optimizer.betas,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs, eta_min=config.scheduler.min_lr
    )

    # 訓練迴圈
    best_val_map = 0
    patience_counter = 0

    for epoch in range(config.training.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"{'='*60}")

        # 訓練
        train_losses = train_epoch(model, train_loader, optimizer, scheduler, config)
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

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mAP": val_metrics["mAP"],
                "config": dict(config),
            }
            torch.save(checkpoint, f"best_model_epoch{epoch}_mAP{val_metrics['mAP']:.4f}.pth")
            print(f"✓ 儲存最佳模型 (mAP: {val_metrics['mAP']:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("\n訓練完成！")
    print(f"最佳 Val mAP: {best_val_map:.4f}")


if __name__ == "__main__":
    main()
