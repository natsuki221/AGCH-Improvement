import sys
from pathlib import Path

def check_python():
    """檢查 Python 版本"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 10, "需要 Python 3.10+"

def check_cuda():
    """檢查 CUDA 可用性"""
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA 不可用 (將使用 CPU，速度會很慢)")
    return cuda_available

def check_packages():
    """檢查關鍵套件"""
    packages = {
        "torch": "2.1.0",
        "transformers": "4.40.0",
        "faiss": "1.7.0",
        "pycocotools": "2.0.0",
    }
    
    for pkg, min_version in packages.items():
        try:
            mod = __import__(pkg)
            version = mod.__version__ if hasattr(mod, "__version__") else "unknown"
            print(f"✓ {pkg:20s} {version}")
        except ImportError:
            print(f"✗ {pkg:20s} 未安裝")
            return False
    
    return True

def check_dataset():
    """檢查資料集"""
    data_dir = Path("./data/coco")
    
    checks = [
        data_dir / "images/train2014",
        data_dir / "images/val2014",
        data_dir / "annotations/instances_train2014.json",
        data_dir / "annotations/captions_train2014.json",
        data_dir / "index_train2014.pkl",
    ]
    
    all_exist = True
    for path in checks:
        if path.exists():
            if path.is_dir():
                n_files = len(list(path.glob("*.jpg")))
                print(f"✓ {path} ({n_files:,} 張影像)")
            else:
                size_mb = path.stat().st_size / 1e6
                print(f"✓ {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {path} 不存在")
            all_exist = False
    
    return all_exist

def check_siglip2():
    """檢查 SigLIP2 模型載入"""
    from transformers import Siglip2Model, Siglip2Processor
    
    try:
        print("\n正在測試 SigLIP2 模型載入...")
        model_name = "google/siglip2-base-patch16-256"
        processor = Siglip2Processor.from_pretrained(model_name)
        model = Siglip2Model.from_pretrained(model_name)
        print(f"✓ SigLIP2 模型載入成功")
        print(f"  參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return True
    except Exception as e:
        print(f"✗ SigLIP2 模型載入失敗: {e}")
        return False

def main():
    print("="*60)
    print("環境驗證")
    print("="*60)
    
    print("\n[1/5] 檢查 Python 版本...")
    check_python()
    
    print("\n[2/5] 檢查 CUDA...")
    cuda_ok = check_cuda()
    
    print("\n[3/5] 檢查 Python 套件...")
    pkg_ok = check_packages()
    
    print("\n[4/5] 檢查資料集...")
    data_ok = check_dataset()
    
    print("\n[5/5] 檢查 SigLIP2 模型...")
    model_ok = check_siglip2()
    
    print("\n" + "="*60)
    if all([cuda_ok, pkg_ok, data_ok, model_ok]):
        print("✅ 環境設置完成！可以開始實驗了。")
    else:
        print("⚠️  部分檢查失敗，請修正後再試。")
    print("="*60)

if __name__ == "__main__":
    main()