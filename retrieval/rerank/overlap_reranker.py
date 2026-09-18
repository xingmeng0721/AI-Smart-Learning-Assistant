"""基于文本重叠的可运行 Reranker：无需外部服务即可跑通链路（开发/测试用）。"""

from typing import List

from retrieval.rerank.base import BaseReranker


def _char_overlap(query: str, text: str) -> float:
    """query 与文本的字符集重叠率，作为轻量相关度信号。"""
    q_set = set(query)
    t_set = set(text)
    if not q_set:
        return 0.0
    inter = q_set & t_set
    return len(inter) / len(q_set)


class OverlapReranker(BaseReranker):
    """按 字符重叠 + 原始排序 加权重排，保序且能提升关键词命中项。"""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        scored = []
        for i, c in enumerate(candidates):
            overlap = _char_overlap(query, c.get("text", ""))
            rank_bonus = 1.0 / (i + 1)  # 原顺序加成
            scored.append((overlap * 0.7 + rank_bonus * 0.3, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [c for _, c in scored]
        if self.top_k:
            return ranked[: self.top_k]
        return ranked
