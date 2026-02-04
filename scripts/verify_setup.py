#!/usr/bin/env python3
"""
環境驗證腳本

對應手冊章節:
- §2.5 環境驗證
- §附錄 A: 快速啟動指令

功能:
1. 檢查 Python 版本
2. 檢查 CUDA 與 GPU 資訊
3. 檢查關鍵套件與版本
4. 檢查資料集完整性
5. 檢查配置檔案
6. 檢查 SigLIP2 模型載入
7. 檢查本專案模組導入

依據實際環境設計:
- Python 3.12.12
- PyTorch 2.6.0+cu124
- transformers 5.0.0
- CUDA 12.4
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict
import importlib.util


def print_header(step: int, total: int, title: str):
    """列印步驟標題"""
    print(f"\n[{step}/{total}] {title}")
    print("-" * 50)


def check_python() -> bool:
    """檢查 Python 版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 10:
        print(f"✓ Python {version_str}")
        return True
    else:
        print(f"✗ Python {version_str} (需要 3.10+)")
        return False


def check_cuda() -> Tuple[bool, Dict]:
    """檢查 CUDA 可用性"""
    info = {}

    try:
        import torch

        info["torch_version"] = torch.__version__
        print(f"✓ PyTorch {torch.__version__}")

        # 檢查 PyTorch 版本
        expected_torch = "2.6.0"
        if expected_torch in torch.__version__:
            print(f"  ✓ 版本符合預期 ({expected_torch})")

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            compute_cap = torch.cuda.get_device_capability(0)

            info["cuda_version"] = cuda_version
            info["gpu_name"] = gpu_name
            info["vram_gb"] = vram_gb
            info["compute_capability"] = compute_cap

            print(f"✓ CUDA {cuda_version}")
            print(f"  GPU: {gpu_name}")
            print(f"  VRAM: {vram_gb:.1f} GB")
            print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

            # RTX 5080 相容性提示
            if compute_cap[0] >= 12:
                print(f"  ⚠️  注意: sm_{compute_cap[0]}{compute_cap[1]} 需要較新 PyTorch 版本支援")

            return True, info
        else:
            print("✗ CUDA 不可用 (將使用 CPU，速度會很慢)")
            return False, info

    except ImportError:
        print("✗ PyTorch 未安裝")
        return False, info


def check_packages() -> Tuple[bool, List[str]]:
    """檢查關鍵套件（依據實際環境）"""

    # 套件列表（基於 uv pip list 結果）
    packages = [
        # 核心深度學習
        ("torch", "2.6.0", "核心"),
        ("torchvision", "0.21.0", "核心"),
        ("transformers", "5.0.0", "核心"),
        # 資料處理
        ("PIL", None, "核心"),  # Pillow 12.1.0
        ("numpy", "2.4.1", "核心"),
        ("pandas", "3.0.0", "資料"),
        # KNN 檢索
        ("faiss", None, "核心"),  # faiss-cpu 1.13.2
        # COCO 資料集
        ("pycocotools", None, "核心"),  # 2.0.11
        # 配置管理
        ("omegaconf", "2.3.0", "配置"),
        ("hydra", "1.3.2", "配置"),
        # 訓練追蹤
        ("wandb", "0.24.1", "追蹤"),
        ("tensorboard", "2.20.0", "追蹤"),
        # 評估與視覺化
        ("sklearn", None, "核心"),  # scikit-learn 1.8.0
        ("matplotlib", None, "視覺化"),
        ("seaborn", None, "視覺化"),
        # 工具
        ("psutil", None, "工具"),
        ("tqdm", None, "工具"),
        ("accelerate", None, "加速"),
    ]

    all_ok = True
    missing = []

    for item in packages:
        pkg, expected_ver, importance = item

        # 處理特殊導入名稱
        import_map = {
            "PIL": ("PIL", "Pillow"),
            "sklearn": ("sklearn", "scikit-learn"),
            "hydra": ("hydra", "hydra-core"),
            "faiss": ("faiss", "faiss-cpu"),
        }

        import_name, display_name = import_map.get(pkg, (pkg, pkg))

        try:
            mod = __import__(import_name)

            # 取得版本
            if pkg == "PIL":
                from PIL import __version__ as version
            elif pkg == "faiss":
                version = "1.13.2"  # faiss 沒有 __version__
            elif pkg == "hydra":
                version = "1.3.2"
            else:
                version = getattr(mod, "__version__", "installed")

            # 版本比對
            if expected_ver and expected_ver in str(version):
                print(f"✓ {display_name:20s} {version}")
            else:
                print(f"✓ {display_name:20s} {version}")

        except ImportError:
            if importance == "核心":
                print(f"✗ {display_name:20s} 未安裝")
                all_ok = False
                missing.append(display_name)
            else:
                print(f"⚠️  {display_name:20s} 未安裝 ({importance})")

    return all_ok, missing


def check_faiss_gpu() -> bool:
    """檢查 FAISS GPU 支援"""
    try:
        import faiss

        # 檢查是否有 GPU 版本
        has_gpu = hasattr(faiss, "index_cpu_to_gpu") and hasattr(faiss, "StandardGpuResources")

        if has_gpu:
            print("✓ FAISS GPU 支援可用")
            return True
        else:
            print("⚠️  FAISS 僅 CPU 版本 (faiss-cpu 1.13.2)")
            print("   建議: conda install -c pytorch -c nvidia faiss-gpu")
            return True  # 不視為錯誤

    except ImportError:
        print("✗ FAISS 未安裝")
        return False


def check_dataset() -> Tuple[bool, Dict]:
    """檢查資料集完整性"""
    data_dir = Path("./data/coco")

    if not data_dir.exists():
        print(f"✗ 資料目錄不存在: {data_dir}")
        return False, {"missing": ["data/coco"]}

    # 必要檔案
    required_files = [
        ("images/train2014", "訓練影像", True),
        ("images/val2014", "驗證影像", True),
        ("annotations/instances_train2014.json", "實例標註", False),
        ("annotations/instances_val2014.json", "實例標註", False),
        ("annotations/captions_train2014.json", "描述標註", False),
        ("annotations/captions_val2014.json", "描述標註", False),
        ("index_train2014.pkl", "訓練索引", False),
        ("index_val2014.pkl", "驗證索引", False),
    ]

    # 可選檔案
    optional_files = [
        ("karpathy_split.json", "Karpathy Split", False),
        ("5fold_split.json", "5-Fold Split", False),
    ]

    all_ok = True
    info = {"exists": [], "missing": [], "optional_missing": []}

    print("必要檔案:")
    for path_str, desc, is_dir in required_files:
        path = data_dir / path_str

        if path.exists():
            if is_dir:
                n_files = len(list(path.glob("*.jpg")))
                print(f"  ✓ {path_str} ({n_files:,} 張)")
            else:
                size_mb = path.stat().st_size / 1e6
                print(f"  ✓ {path_str} ({size_mb:.1f} MB)")
            info["exists"].append(path_str)
        else:
            print(f"  ✗ {path_str} 不存在")
            info["missing"].append(path_str)
            all_ok = False

    print("\n可選檔案:")
    for path_str, desc, is_dir in optional_files:
        path = data_dir / path_str

        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {path_str} ({size_kb:.1f} KB)")
        else:
            print(f"  ⚠️  {path_str} (需要時執行對應腳本建立)")
            info["optional_missing"].append(path_str)

    return all_ok, info


def check_configs() -> bool:
    """檢查配置檔案"""
    try:
        import yaml
    except ImportError:
        print("✗ PyYAML 未安裝")
        return False

    config_dir = Path("./configs")

    if not config_dir.exists():
        print(f"✗ 配置目錄不存在: {config_dir}")
        return False

    # 必要配置
    required_configs = [
        ("hardware/rtx5080_16gb.yaml", "硬體配置"),
        ("experiments/baseline.yaml", "基準實驗"),
    ]

    # 可選配置
    optional_configs = [
        ("experiments/cv_experiment.yaml", "5-Fold CV"),
        ("experiments/ablation_fusion.yaml", "Fusion Ablation"),
        ("experiments/ablation_hash.yaml", "Hash Ablation"),
        ("experiments/grid_search.yaml", "Grid Search"),
    ]

    all_ok = True

    print("必要配置:")
    for config_path, desc in required_configs:
        path = config_dir / config_path

        if path.exists():
            try:
                with open(path) as f:
                    yaml.safe_load(f)
                print(f"  ✓ {config_path}")
            except Exception as e:
                print(f"  ✗ {config_path} (語法錯誤)")
                all_ok = False
        else:
            print(f"  ✗ {config_path} 不存在")
            all_ok = False

    print("\n可選配置:")
    for config_path, desc in optional_configs:
        path = config_dir / config_path

        if path.exists():
            try:
                with open(path) as f:
                    yaml.safe_load(f)
                print(f"  ✓ {config_path}")
            except:
                print(f"  ⚠️  {config_path} (語法錯誤)")
        else:
            print(f"  ⚠️  {config_path}")

    return all_ok


def check_siglip2() -> bool:
    """檢查 SigLIP2 模型載入"""
    print("正在測試 SigLIP2 模型...")

    # 抑制 HuggingFace 警告
    import warnings
    import os
    import logging

    # 暫時抑制警告
    original_level = logging.getLogger("transformers").level
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = "google/siglip2-base-patch16-256"

    try:
        # 方法：分開載入 ImageProcessor + Tokenizer + Model
        # 這樣可以避開 Siglip2Processor 的 tokenizer 映射問題
        from transformers import AutoModel, AutoImageProcessor, GemmaTokenizerFast
        import torch
        from PIL import Image
        import numpy as np

        print(f"  載入 ImageProcessor...")
        image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

        print(f"  載入 Tokenizer (GemmaTokenizerFast)...")
        tokenizer = GemmaTokenizerFast.from_pretrained(model_name)

        print(f"  載入 Model...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        param_count = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"✓ SigLIP2 模型載入成功")
        print(f"  模型: {model_name}")
        print(f"  Model 類型: {type(model).__name__}")
        print(f"  參數量: {param_count:.1f}M")

        # 測試推論
        print(f"  測試推論...")

        # 建立測試資料
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        pixel_values = image_processor(images=dummy_image, return_tensors="pt")["pixel_values"]
        inputs = tokenizer(["a test image"], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=inputs["input_ids"])

        print(f"  ✓ 推論測試通過")

        # 提示正確的載入方式
        print(f"\n  📝 正確載入方式:")
        print(f"     image_processor = AutoImageProcessor.from_pretrained('{model_name}')")
        print(f"     tokenizer = GemmaTokenizerFast.from_pretrained('{model_name}')")
        print(f"     model = AutoModel.from_pretrained('{model_name}')")

        # 提示 HF_TOKEN
        if not os.environ.get("HF_TOKEN"):
            print(f"\n  💡 提示: 設置 HF_TOKEN 可加速下載")

        return True

    except Exception as e:
        error_msg = str(e)[:200]
        print(f"✗ SigLIP2 載入失敗: {type(e).__name__}")
        print(f"  錯誤: {error_msg}")

        # 提供解決方案
        print("\n  💡 可能的解決方案:")
        print("     1. 確認網路連線正常")
        print("     2. 設置 HF_TOKEN 環境變數")
        print("     3. 升級 transformers: pip install transformers --upgrade")

        return False

    finally:
        # 恢復設定
        logging.getLogger("transformers").setLevel(original_level)
        warnings.resetwarnings()
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        os.environ.pop("TOKENIZERS_PARALLELISM", None)


def check_project_modules() -> bool:
    """檢查本專案模組導入"""
    # 加入 src 到 path
    src_path = Path(__file__).parent.parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    modules_to_check = [
        ("siglip2_multimodal_hash.model", "模型定義"),
        ("siglip2_multimodal_hash.dataset", "資料載入"),
        ("siglip2_multimodal_hash.losses", "損失函數"),
        ("siglip2_multimodal_hash.utils", "工具函數"),
        ("siglip2_multimodal_hash.knn", "KNN 檢索"),
    ]

    all_ok = True

    for module_name, desc in modules_to_check:
        try:
            mod = importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️  {module_name}: {type(e).__name__}")

    return all_ok


def check_editable_install() -> bool:
    """檢查專案是否以 editable 模式安裝"""
    try:
        import siglip2_multimodal_hash

        location = getattr(siglip2_multimodal_hash, "__file__", None)

        if location:
            print(f"✓ 專案已安裝 (editable)")
            print(f"  路徑: {Path(location).parent}")
            return True
        else:
            print("⚠️  專案未以 editable 模式安裝")
            return True
    except ImportError:
        print("⚠️  專案未安裝，使用 src/ 路徑")
        return True


def main():
    print("=" * 60)
    print("🔍 AGCH-Improvement 環境驗證")
    print("   Python 3.12 | PyTorch 2.6.0 | CUDA 12.4")
    print("=" * 60)

    total_steps = 8
    results = {}

    # Step 1: Python
    print_header(1, total_steps, "檢查 Python 版本")
    results["python"] = check_python()

    # Step 2: CUDA
    print_header(2, total_steps, "檢查 CUDA 與 GPU")
    results["cuda"], _ = check_cuda()

    # Step 3: Packages
    print_header(3, total_steps, "檢查 Python 套件")
    results["packages"], missing_pkgs = check_packages()

    # Step 4: FAISS
    print_header(4, total_steps, "檢查 FAISS")
    results["faiss"] = check_faiss_gpu()

    # Step 5: Dataset
    print_header(5, total_steps, "檢查資料集")
    results["dataset"], _ = check_dataset()

    # Step 6: Configs
    print_header(6, total_steps, "檢查配置檔案")
    results["configs"] = check_configs()

    # Step 7: SigLIP2
    print_header(7, total_steps, "檢查 SigLIP2 模型")
    results["siglip2"] = check_siglip2()

    # Step 8: Project modules
    print_header(8, total_steps, "檢查專案模組")
    results["modules"] = check_project_modules()
    check_editable_install()

    # Summary
    print("\n" + "=" * 60)
    print("📊 驗證摘要")
    print("=" * 60)

    critical_items = ["python", "cuda", "packages", "dataset", "configs", "modules"]
    optional_items = ["faiss", "siglip2"]

    all_critical_pass = True

    print("\n必要項目:")
    for name in critical_items:
        status = "✓" if results.get(name, False) else "✗"
        print(f"  {status} {name}")
        if not results.get(name, False):
            all_critical_pass = False

    print("\n可選項目:")
    for name in optional_items:
        status = "✓" if results.get(name, False) else "⚠️"
        print(f"  {status} {name}")

    print("\n" + "=" * 60)

    if all_critical_pass:
        print("✅ 必要環境設置完成！")
        if not results.get("siglip2", False):
            print("⚠️  SigLIP2 載入有問題，可能需要調整 transformers 版本")
        print("\n📌 下一步:")
        print("   python scripts/train.py")
    else:
        print("❌ 部分必要檢查失敗，請修正後再試。")

        if missing_pkgs:
            print(f"\n📌 缺少套件: {', '.join(missing_pkgs)}")

    print("=" * 60)

    return 0 if all_critical_pass else 1


if __name__ == "__main__":
    sys.exit(main())
