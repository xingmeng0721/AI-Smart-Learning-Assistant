"""LLM 客户端抽象。"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, List


class BaseLLM(ABC):
    @abstractmethod
    async def complete(self, messages: List[dict], **kwargs) -> str:
        """非流式完整生成。messages 形如 [{'role':'system'|'user'|'assistant','content':...}]"""

    @abstractmethod
    def stream(self, messages: List[dict], **kwargs) -> AsyncIterator[str]:
        """流式生成，逐段产出文本 token。"""
