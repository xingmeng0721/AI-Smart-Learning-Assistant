"""确定性本地哈希 Embedding：无需外部服务即可跑通全链路，用于开发/测试。"""

import hashlib
import math
from typing import List

from embedding.base import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """基于字符 n-gram 哈希的确定性向量，便于离线验证 RAG 链路。"""

    def __init__(self, dim: int = 256):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        # 字符 + 2-gram + 3-gram 特征哈希
        grams = list(text)
        for n in (2, 3):
            grams.extend(text[i : i + n] for i in range(max(0, len(text) - n + 1)))
        for g in grams:
            h = hashlib.md5(g.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
