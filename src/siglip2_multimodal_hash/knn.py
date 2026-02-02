# src/siglip2_multimodal_hash/knn.py
"""
KNN 檢索模組

對應手冊章節:
- §6.1 建立 Hash Index
- §6.2 KNN 檢索與投票

提供 Hash-based KNN 檢索功能，支援:
1. FAISS Binary Index 建立與管理
2. 加權投票策略（softmax/uniform/rank-based）
3. Top-N 標籤預測
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import pickle

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS 未安裝，KNN 功能將無法使用")


class HashIndex:
    """
    Hash Index 封裝類別

    管理 FAISS binary index 的建立、儲存與搜尋
    """

    def __init__(self, hash_bits: int, use_gpu: bool = False):
        """
        Args:
            hash_bits: Hash 碼位數
            use_gpu: 是否使用 GPU 加速搜尋
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS 未安裝，請執行: pip install faiss-cpu 或 faiss-gpu")

        self.hash_bits = hash_bits
        self.use_gpu = use_gpu

        # 建立空的 binary index
        self.index = faiss.IndexBinaryFlat(hash_bits)

        # 儲存對應的標籤
        self.labels: Optional[np.ndarray] = None

        # 儲存對應的 image IDs（用於可解釋性）
        self.image_ids: Optional[List[int]] = None

        # GPU 加速
        if use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("✓ FAISS GPU 加速已啟用")

    def add(
        self, hash_codes: np.ndarray, labels: np.ndarray, image_ids: Optional[List[int]] = None
    ):
        """
        添加 hash codes 到索引

        Args:
            hash_codes: (N, hash_bits) float tensor，會自動二值化
            labels: (N, num_classes) multi-hot labels
            image_ids: 可選的 image ID 列表
        """
        # 二值化（soft hash -> hard binary）
        binary_codes = (hash_codes > 0).astype(np.uint8)

        # 打包成 FAISS 需要的格式
        # FAISS binary index 需要 (N, hash_bits/8) uint8 array
        packed_codes = np.packbits(binary_codes, axis=1)

        self.index.add(packed_codes)

        # 儲存標籤
        if self.labels is None:
            self.labels = labels
        else:
            self.labels = np.vstack([self.labels, labels])

        # 儲存 image IDs
        if image_ids is not None:
            if self.image_ids is None:
                self.image_ids = image_ids
            else:
                self.image_ids.extend(image_ids)

    def search(self, query_hash: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜尋最近鄰

        Args:
            query_hash: (B, hash_bits) float tensor
            k: 返回的鄰居數量

        Returns:
            distances: (B, k) Hamming distances
            indices: (B, k) 鄰居索引
        """
        # 二值化
        binary_codes = (query_hash > 0).astype(np.uint8)
        packed_codes = np.packbits(binary_codes, axis=1)

        # 搜尋
        distances, indices = self.index.search(packed_codes, k)

        return distances, indices

    @property
    def ntotal(self) -> int:
        """索引中的總樣本數"""
        return self.index.ntotal

    def save(self, path: Union[str, Path]):
        """
        儲存索引到檔案

        Args:
            path: 儲存路徑（不含副檔名）
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 如果是 GPU index，轉回 CPU 再儲存
        cpu_index = self.index
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)

        # 儲存 FAISS index
        faiss.write_index_binary(cpu_index, str(path.with_suffix(".index")))

        # 儲存 labels 與 image_ids
        metadata = {"labels": self.labels, "image_ids": self.image_ids, "hash_bits": self.hash_bits}
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"✓ Hash Index 已儲存: {path}")
        print(f"  - 樣本數: {self.ntotal:,}")
        print(f"  - Hash bits: {self.hash_bits}")

    @classmethod
    def load(cls, path: Union[str, Path], use_gpu: bool = False) -> "HashIndex":
        """
        從檔案載入索引

        Args:
            path: 儲存路徑（不含副檔名）
            use_gpu: 是否使用 GPU 加速

        Returns:
            HashIndex 實例
        """
        path = Path(path)

        # 載入 metadata
        with open(path.with_suffix(".meta"), "rb") as f:
            metadata = pickle.load(f)

        # 建立實例
        instance = cls(hash_bits=metadata["hash_bits"], use_gpu=False)  # 先用 CPU 載入

        # 載入 FAISS index
        instance.index = faiss.read_index_binary(str(path.with_suffix(".index")))
        instance.labels = metadata["labels"]
        instance.image_ids = metadata["image_ids"]

        # 如果需要 GPU，轉換
        if use_gpu and hasattr(faiss, "index_cpu_to_gpu"):
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpu(res, 0, instance.index)
            instance.use_gpu = True
            print("✓ FAISS GPU 加速已啟用")

        print(f"✓ Hash Index 已載入: {path}")
        print(f"  - 樣本數: {instance.ntotal:,}")
        print(f"  - Hash bits: {instance.hash_bits}")

        return instance


def predict_tags(
    query_hash: np.ndarray,
    index: HashIndex,
    k: int = 20,
    tau: float = 0.07,
    voting_strategy: str = "softmax",
    top_n: int = 5,
    return_neighbors: bool = False,
) -> Dict[str, np.ndarray]:
    """
    KNN 加權投票預測標籤

    對應手冊 §6.2 KNN 檢索與投票

    Args:
        query_hash: (B, hash_bits) float tensor 或 np.ndarray
        index: HashIndex 實例
        k: 鄰居數量
        tau: softmax 溫度參數
        voting_strategy: 投票策略 ("softmax", "uniform", "rank_based")
        top_n: 返回的 Top-N 標籤數量
        return_neighbors: 是否返回鄰居資訊（用於可解釋性）

    Returns:
        字典包含:
        - tag_scores: (B, num_classes) 每個類別的投票分數
        - top_indices: (B, top_n) Top-N 類別索引
        - top_scores: (B, top_n) Top-N 分數
        - neighbor_indices: (B, k) 鄰居索引（如果 return_neighbors=True）
        - neighbor_distances: (B, k) 鄰居距離（如果 return_neighbors=True）
    """
    # 確保是 numpy array
    if isinstance(query_hash, torch.Tensor):
        query_hash = query_hash.cpu().numpy()

    batch_size = query_hash.shape[0]
    num_classes = index.labels.shape[1]

    # 1. KNN 搜尋
    distances, indices = index.search(query_hash, k)

    # 2. 計算相似度（Hamming distance -> similarity）
    # Hamming distance 範圍是 [0, hash_bits]
    # similarity 範圍是 [0, 1]
    similarities = 1 - distances / index.hash_bits

    # 3. 計算權重
    if voting_strategy == "softmax":
        # Softmax 加權，溫度參數控制平滑度
        weights = np.exp(similarities / tau)
        weights = weights / weights.sum(axis=1, keepdims=True)
    elif voting_strategy == "uniform":
        # 均勻權重
        weights = np.ones_like(similarities) / k
    elif voting_strategy == "rank_based":
        # 基於排名的權重（排名越前越重）
        ranks = np.arange(1, k + 1).astype(float)
        weights = 1 / ranks[np.newaxis, :]
        weights = weights / weights.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"未知的投票策略: {voting_strategy}")

    # 4. 加權投票
    # neighbor_labels: (B, k, num_classes)
    neighbor_labels = index.labels[indices]

    # tag_scores: (B, num_classes) = sum over k of (weight * label)
    tag_scores = np.einsum("bk,bkc->bc", weights, neighbor_labels)

    # 5. Top-N 選擇
    # 對每個樣本取分數最高的 top_n 個類別
    top_indices = np.argsort(tag_scores, axis=1)[:, -top_n:][:, ::-1]
    top_scores = np.take_along_axis(tag_scores, top_indices, axis=1)

    result = {"tag_scores": tag_scores, "top_indices": top_indices, "top_scores": top_scores}

    if return_neighbors:
        result["neighbor_indices"] = indices
        result["neighbor_distances"] = distances
        if index.image_ids is not None:
            # 獲取鄰居的 image IDs
            neighbor_image_ids = [[index.image_ids[i] for i in row] for row in indices]
            result["neighbor_image_ids"] = neighbor_image_ids

    return result


def compute_knn_metrics(
    predictions: Dict[str, np.ndarray], labels: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    計算 KNN 預測的評估指標

    Args:
        predictions: predict_tags() 的輸出
        labels: (B, num_classes) 真實標籤
        threshold: 分類閾值

    Returns:
        包含各項指標的字典
    """
    tag_scores = predictions["tag_scores"]

    # 二值化預測
    y_pred = (tag_scores > threshold).astype(int)
    y_true = labels.astype(int)

    # Precision, Recall, F1
    tp = np.sum(y_pred * y_true, axis=1)
    fp = np.sum(y_pred * (1 - y_true), axis=1)
    fn = np.sum((1 - y_pred) * y_true, axis=1)

    precision = np.mean(tp / (tp + fp + 1e-8))
    recall = np.mean(tp / (tp + fn + 1e-8))
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # 準確率（Exact Match）
    exact_match = np.mean(np.all(y_pred == y_true, axis=1))

    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": exact_match}
