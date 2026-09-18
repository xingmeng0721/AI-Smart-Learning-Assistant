"""Query Rewrite：多轮指代消解与多意图拆分。"""

import json
import re
from typing import List

from llm.base import BaseLLM

_SYSTEM_PROMPT = (
    "你负责把用户在多轮对话中可能带有指代、省略的提问，改写为可独立检索的清晰检索语句。"
    "要求：\n"
    "1. 若原问题包含多个可拆分的意图，用 JSON 数组输出拆分后的多个检索语句；\n"
    "2. 若只有单一意图，输出包含单个语句的 JSON 数组；\n"
    "3. 只输出 JSON，不要多余解释。"
)


class QueryRewriter:
    def __init__(self, llm: BaseLLM, max_history: int = 4):
        self.llm = llm
        self.max_history = max_history

    async def rewrite(self, current_query: str, history: List[str]) -> List[str]:
        """改写并拆分，返回一个或多个检索语句。无 LLM 时退化为原句。"""
        if not history:
            return [current_query]
        history_text = "\n".join(f"用户: {q}" for q in history[-self.max_history :])
        user_prompt = (
            f"历史对话:\n{history_text}\n\n"
            f"当前提问: {current_query}\n"
            "请输出改写后的检索语句 JSON 数组。"
        )
        try:
            raw = await self.llm.complete(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            parsed = self._parse_json_array(raw)
            if parsed:
                return parsed
        except Exception:
            pass
        return [current_query]

    @staticmethod
    def _parse_json_array(raw: str) -> List[str]:
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
