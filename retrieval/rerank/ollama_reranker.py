"""Ollama Rerank 客户端：调用 /api/rerank 对候选二次精排。"""

from typing import List, Optional

import httpx

from retrieval.rerank.base import BaseReranker


class OllamaReranker(BaseReranker):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-reranker-v2-m3",
        top_k: Optional[int] = None,
        timeout: int = 120,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_k = top_k
        self.timeout = timeout
        self._transport = transport

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        if not candidates:
            return []
        documents = [c.get("text", "") for c in candidates]
        payload = {"model": self.model, "query": query, "documents": documents}
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            resp = client.post(f"{self.base_url}/api/rerank", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama rerank 失败: HTTP {resp.status_code}: {resp.text[:200]}")
            results = resp.json().get("results", [])

        # 按 relevance_score 降序
        results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        ranked = []
        for r in results:
            idx = r.get("index", 0)
            if idx >= len(candidates):
                continue
            c = dict(candidates[idx])
            c["score"] = float(r.get("relevance_score", 0.0))
            ranked.append(c)
        if self.top_k:
            return ranked[: self.top_k]
        return ranked
