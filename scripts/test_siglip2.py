import torch
from transformers import Siglip2Model, Siglip2Processor
from PIL import Image
import requests
from io import BytesIO

# 載入模型
print("載入 SigLIP2 模型...")
model_name = "google/siglip2-base-patch16-256"
processor = Siglip2Processor.from_pretrained(model_name)
model = Siglip2Model.from_pretrained(model_name)

# 移到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"模型已載入到 {device}")

# 下載測試影像
print("\n下載測試影像...")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# 準備輸入
text = "Two cats sleeping on a couch"
inputs = processor(
    text=[text],
    images=image,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 前向傳播
print("\n執行前向傳播...")
with torch.no_grad():
    outputs = model(**inputs)

# 檢查輸出
image_embeds = outputs.image_embeds  # (1, D)
text_embeds = outputs.text_embeds    # (1, D)

print(f"✓ 影像 embedding shape: {image_embeds.shape}")
print(f"✓ 文字 embedding shape: {text_embeds.shape}")

# 計算相似度
similarity = torch.cosine_similarity(image_embeds, text_embeds)
print(f"✓ 圖文相似度: {similarity.item():.4f}")

print("\n✅ SigLIP2 測試通過！")