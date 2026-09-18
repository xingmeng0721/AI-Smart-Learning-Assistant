"""检索评测指标：用于 P10 量化"混合检索 + Rerank"相对单路向量的提升。"""

from typing import Iterable, List, Sequence, Union

Gold = Union[Iterable[str], str]


def _as_set(gold: Gold) -> set:
    if isinstance(gold, str):
        return {gold}
    return set(gold)


def rank_first(ranked_ids: Sequence[str], gold: Gold) -> int:
    """返回第一个命中 gold 的 1-based rank；未命中返回 0。"""
    target = _as_set(gold)
    for i, cid in enumerate(ranked_ids):
        if cid in target:
            return i + 1
    return 0


def recall_at_k(ranked_ids: Sequence[str], gold: Gold, k: int) -> float:
    """单条查询的 Recall@k：gold 是否出现在前 k 位（布尔值转 0/1）。"""
    r = rank_first(ranked_ids, gold)
    return 1.0 if 0 < r <= k else 0.0


def mrr(ranked_ids: Sequence[str], gold: Gold) -> float:
    """单条查询的 MRR：第一个命中 gold 位置的倒数；未命中为 0。"""
    r = rank_first(ranked_ids, gold)
    return 1.0 / r if r else 0.0