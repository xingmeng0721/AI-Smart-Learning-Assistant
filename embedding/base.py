"""Embedding 抽象接口。"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """向量化接口：所有实现（本地/Ollama/其他）需满足此契约。"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量返回文本向量。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
