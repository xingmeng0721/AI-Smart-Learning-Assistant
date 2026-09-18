"""多路检索结果融合：RRF 与加权分数融合。"""

from typing import Dict, List


def rrf_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
    weights: List[float] | None = None,
) -> List[str]:
    """Reciprocal Rank Fusion：多路按 rank 求和，返回排序后的 chunk_id 列表。

    ranked_lists 元素形如 [{'chunk_id':..,'rank':0}, ...]（rank 从 0 起）。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    acc: Dict[str, float] = {}
    for rlist, w in zip(ranked_lists, weights):
        for item in rlist:
            cid = item["chunk_id"]
            rank = item.get("rank", 0) + 1  # 转 1-based
            acc[cid] = acc.get(cid, 0.0) + w * (1.0 / (k + rank))
    return sorted(acc, key=acc.get, reverse=True)


def score_fusion(
    ranked_lists: List[List[dict]],
    weights: List[float] | None = None,
) -> List[str]:
    """加权分数融合：对每路归一化 score 加权求和，返回排序后的 chunk_id 列表。"""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    acc: Dict[str, float] = {}
    for rlist, w in zip(ranked_lists, weights):
        if not rlist:
            continue
        max_s = max(item.get("score", 0.0) for item in rlist) or 1.0
        for item in rlist:
            cid = item["chunk_id"]
            norm = (item.get("score", 0.0) or 0.0) / max_s
            acc[cid] = acc.get(cid, 0.0) + w * norm
    return sorted(acc, key=acc.get, reverse=True)
