"""开发/测试用 Mock LLM：无 Ollama 时跑通全链路。"""

import asyncio
from typing import AsyncIterator, List

from llm.base import BaseLLM


class MockLLM(BaseLLM):
    """基于规则生成可预测的回复，便于验证链路与测试。"""

    def __init__(self, on_rewrite=None, on_answer=None):
        # 回调可注入自定义行为
        self.on_rewrite = on_rewrite
        self.on_answer = on_answer

    def _default_answer(self, messages: List[dict]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"（Mock）针对「{last_user}」的检索式回答。"

    async def complete(self, messages: List[dict], **kwargs) -> str:
        return self._default_answer(messages)

    async def stream(self, messages: List[dict], **kwargs) -> AsyncIterator[str]:
        text = await self.complete(messages, **kwargs)
        # 逐字产出，模拟流式
        for i in range(0, len(text), 2):
            yield text[i : i + 2]
            await asyncio.sleep(0)
