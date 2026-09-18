"""置信度评估与引用溯源：幻觉控制核心。"""

from typing import List


class ConfidenceEvaluator:
    """基于检索分数评估 Query 是否落在知识库边界内。"""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def evaluate(self, candidates: List[dict]) -> dict:
        """返回 {'in_kb': bool, 'confidence': float, 'top_score': float}。

        置信度取融合后候选最高相似度；低于阈值判定为知识库外。
        candidates 需包含 'score' 字段（0~1 相似度）。
        """
        if not candidates:
            return {"in_kb": False, "confidence": 0.0, "top_score": 0.0}
        top = max(float(c.get("score", 0.0) or 0.0) for c in candidates)
        # 有候选但分数低 → 边界外；无候选 → 边界外
        in_kb = top >= self.threshold
        return {"in_kb": in_kb, "confidence": top, "top_score": top}


def build_citations(candidates: List[dict]) -> List[dict]:
    """从候选生成引用溯源：绑定原始 Chunk Metadata。"""
    out: List[dict] = []
    for c in candidates:
        meta = c.get("metadata", {}) or {}
        out.append(
            {
                "chunk_id": c.get("chunk_id", ""),
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "section_path": meta.get("section_path", ""),
                "source": meta.get("source", ""),
                "page": meta.get("page", ""),
            }
        )
    return out
