"""基于 BAAI/bge-reranker-v2-m3 的 Cross-Encoder 精排器（本地 HF 模型）。

- 首次调用时惰性加载模型（约 2.3GB，CPU）。
- 模型路径可指向本地目录（推荐放 D 盘），或 HF 仓库名自动下载。
- 将 raw relevance logit 经 sigmoid 归一化为 (0,1)，与置信度阈值尺度兼容。
"""

import math
from typing import List, Optional

from retrieval.rerank.base import BaseReranker


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class BgeReranker(BaseReranker):
    def __init__(
        self,
        model_path: str = "BAAI/bge-reranker-v2-m3",
        top_k: Optional[int] = None,
        use_fp16: bool = False,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_path = model_path
        self.top_k = top_k
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from FlagEmbedding import FlagReranker  # noqa: PLC0415
        except ImportError as ex:  # pragma: no cover - 依赖缺失路径
            raise RuntimeError(
                "FlagEmbedding 未安装，无法使用 BgeReranker（pip install FlagEmbedding）"
            ) from ex
        self._model = FlagReranker(self.model_path, use_fp16=self.use_fp16)

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        texts = [c.get("text", "") for c in candidates]
        self._load()
        pairs = [(query, t) for t in texts]
        scores = self._model.compute_score(pairs, batch_size=self.batch_size)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for c, s in zip(candidates, scores):
            raw = float(s)
            c["score"] = _sigmoid(raw) if self.normalize else raw
        ranked = sorted(
            candidates, key=lambda c: c.get("score", 0.0), reverse=True
        )
        if self.top_k:
            return ranked[: self.top_k]
        return ranked