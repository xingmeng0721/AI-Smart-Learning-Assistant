"""Rerank 抽象接口。"""

from abc import ABC, abstractmethod
from typing import List


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[dict]) -> List[dict]:
        """对候选重新排序。candidates 为检索结果 dict（含 chunk_id/text/metadata）。
        返回按相关度降序的新列表。"""
