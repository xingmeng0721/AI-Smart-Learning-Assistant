"""生成链路：组装 Prompt、注入引用上下文、拒答判定、流式生成。"""

from typing import AsyncIterator, Dict, List

from llm.base import BaseLLM
from retrieval.scoring import build_citations


def build_context(candidates: List[dict]) -> str:
    """将候选拼成带 [来源N] 标记的上下文。"""
    parts = []
    for i, c in enumerate(candidates, 1):
        meta = c.get("metadata", {}) or {}
        section = meta.get("section_path") or meta.get("title") or "未知章节"
        parts.append(f"[来源{i}]（{section}）\n{c.get('text', '')}")
    return "\n\n".join(parts)


def build_citations_markdown(candidates: List[dict]) -> str:
    """引用列表（Markdown 格式）。"""
    out = []
    for i, c in enumerate(candidates, 1):
        meta = c.get("metadata", {}) or {}
        src = meta.get("source") or meta.get("doc_id") or "?"
        section = meta.get("section_path") or meta.get("title") or "?"
        page = meta.get("page") or ""
        page_s = f" · 第{page}页" if page else ""
        out.append(f"{i}. [{section}]({src}){page_s} — {c.get('chunk_id','')}")
    return "\n".join(out)


class Generator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _messages(
        self,
        rewritten_queries: List[str],
        context: str,
        citations: str,
        history: List[dict],
    ) -> List[dict]:
        system = (
            "你是 AI 智能学习助教。请严格依据下面的参考资料回答，并遵循以下规则：\n"
            "1) 只输出答案正文，引用处用 [来源N] 标注；\n"
            "2) 答案写完后立即结束，不要复述、重列或解释引用，也不要输出『引用』『來源』『参考资料』等多余段落或符号；\n"
            "3) 若资料不足以回答问题，直接说明无法回答，绝不编造；\n"
            "4) 不要输出与问题无关的内容；\n"
            "5) 回答应尽量完整、详细、深入：先直接作答，再用分点、段落或示例充分展开原理、原因与要点，"
            "结合资料把内容讲透，避免只给一句话或简单罗列。\n\n"
            f"参考资料：\n{context or '（无）'}\n\n"
            f"出处映射（仅供你检索 [来源N] 对应关系，禁止输出）：\n{citations or '（无）'}"
        )
        msgs = [{"role": "system", "content": system}]
        # 历史（不含当前提问）
        msgs.extend(history[-4:])
        user_content = "\n".join(
            [f"检索意图{i+1}: {q}" for i, q in enumerate(rewritten_queries)]
        )
        msgs.append({"role": "user", "content": user_content or "请回答。"})
        return msgs

    async def generate(
        self,
        rewritten_queries: List[str],
        context: str,
        citations: str,
        history: List[dict],
        **kwargs,
    ) -> str:
        msgs = self._messages(rewritten_queries, context, citations, history)
        return await self.llm.complete(msgs, **kwargs)

    def generate_stream(
        self,
        rewritten_queries: List[str],
        context: str,
        citations: str,
        history: List[dict],
        **kwargs,
    ) -> AsyncIterator[str]:
        msgs = self._messages(rewritten_queries, context, citations, history)
        return self.llm.stream(msgs, **kwargs)
