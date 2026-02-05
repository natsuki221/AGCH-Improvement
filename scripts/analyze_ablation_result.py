import torch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def analyze_checkpoint(path):
    print(f"Loading checkpoint: {path}")
    if not Path(path).exists():
        print(f"Checkpoint not found at {path}!")
        return

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("val_metrics", {})

        print("\n" + "=" * 40)
        print("Experiment Results Analysis")
        print("=" * 40)

        # Key metrics
        print(f"mAP: {metrics.get('mAP', 0):.4f}")
        print(f"AUC (Macro): {metrics.get('auc_macro', 0):.4f}")
        print(f"F1 (Macro): {metrics.get('f1_macro', 0):.4f}")
        print(f"Precision (Macro): {metrics.get('precision_macro', 0):.4f}")
        print(f"Recall (Macro): {metrics.get('recall_macro', 0):.4f}")
        print(f"Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        print(f"LRAP: {metrics.get('lrap', 0):.4f}")

        print("-" * 40)
        print("Full Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


if __name__ == "__main__":
    analyze_checkpoint("outputs/checkpoints/ablation_no_hash/best_model.pth")
