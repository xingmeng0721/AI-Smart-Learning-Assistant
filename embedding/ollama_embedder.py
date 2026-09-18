"""Ollama Embedding 客户端：调用 /api/embed 分批向量化，含失败重试。"""

import time
from typing import List, Optional

import httpx

from embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        timeout: int = 120,
        max_attempts: int = 3,
        batch_size: int = 16,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.batch_size = batch_size
        self._transport = transport
        self._dim = 0

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one_batch(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": list(texts)}
        attempt = 0
        while True:
            try:
                with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                    resp = client.post(f"{self.base_url}/api/embed", json=payload)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Ollama embed 失败: HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json()
                return data.get("embeddings", [])
            except (httpx.TransportError, RuntimeError):
                # 偶发的 Ollama tokenizer 运行器连接抖动：短退避重试
                attempt += 1
                if attempt >= self.max_attempts:
                    raise
                time.sleep(0.5 * attempt)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """分批向量化：按 batch_size 切片分段请求，避免单次请求过大。

        每批独立失败重试；拼接返回所有向量。
        """
        if not texts:
            return []
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_one_batch(batch)
            if vecs:
                self._dim = len(vecs[0])
            all_vecs.extend(vecs)
        return all_vecs
