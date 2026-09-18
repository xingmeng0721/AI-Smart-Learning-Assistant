"""Agent 可用工具：笔记查询 / 资料检索 / 章节跳转。

工具基于 HybridRetriever 实现，返回格式化文本片段供 LLM 二次加工。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

from agent.schemas import ParamSpec, ToolDef
from retrieval.hybrid import HybridRetriever
from retrieval.tfidf_index import TfidfIndex


def _fmt(result: dict, src: str = "") -> str:
    meta = result.get("metadata", {}) or {}
    section = meta.get("section_path") or meta.get("title") or "未知章节"
    head = f"[{section}]({src or meta.get('source', '')})"
    return f"{head}\n{result.get('text', '')}"


@dataclass
class Tool:
    """可执行工具：描述 + 可调用函数。"""

    name: str
    description: str
    parameters: List[ParamSpec]
    func: Callable[..., str]


class AgentTools:
    """把 HybridRetriever 包装成语义/关键词/章节三种工具，并生成工具集。"""

    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever
        # 建立 chunk_id -> 全量结果的映射，供关键词检索与章节跳转取文本
        self._by_id: Dict[str, dict] = {
            it["chunk_id"]: it for it in retriever._store.all_items()
        }

    # ---------- 工具实现 ----------
    def query_notes(self, query: str = "", top_k: int = 5) -> str:
        """语义/混合检索笔记内容，返回最相关片段。"""
        if not query:
            return "（query_notes 需要提供 query 参数）"
        top_k = max(1, min(int(top_k), 10))
        cands = self._retriever.retrieve(query, top_k=top_k)
        if not cands:
            return "（未检索到相关内容）"
        return "\n\n".join(_fmt(c, "query_notes") for c in cands)

    def search_material(self, keyword: str = "", top_k: int = 5) -> str:
        """按关键词做 TF-IDF 精确检索辅助材料，返回命中片段。"""
        if not keyword:
            return "（search_material 需要提供 keyword 参数）"
        top_k = max(1, min(int(top_k), 10))
        ids = [
            d["chunk_id"]
            for d in self._retriever._tfidf.query(keyword, top_k=top_k)
            if d.get("score", 0.0) > 1e-9  # 过滤零相似度（无词面重叠）
        ]
        hits = [self._by_id[cid] for cid in ids if cid in self._by_id]
        if not hits:
            return "（未检索到相关材料）"
        return "\n\n".join(_fmt(h, "search_material") for h in hits)

    def jump_chapter(self, section: str = "") -> str:
        """跳转到指定章节，返回该章节完整正文。"""
        if not section:
            return "（jump_chapter 需要提供 section 参数）"
        for it in self._by_id.values():
            meta = it.get("metadata", {}) or {}
            hay = f"{meta.get('section_path','')}|{meta.get('title','')}"
            if section in hay:
                return _fmt(it, "jump_chapter")
        return f"（未找到章节：{section}）"

    # ---------- 工具集 ----------
    def tools(self) -> List[Tool]:
        return [
            Tool(
                name="query_notes",
                description="在知识库中做语义检索，返回与问题最相关的笔记片段，适合回答具体知识问题。",
                parameters=[
                    ParamSpec("query", "string", "用户问题或改写后的查询", True),
                    ParamSpec("top_k", "integer", "返回片段数，默认 5", False),
                ],
                func=self.query_notes,
            ),
            Tool(
                name="search_material",
                description="按关键词做精确检索，定位名称/术语相关的辅助材料，适合核对专有名词。",
                parameters=[
                    ParamSpec("keyword", "string", "想检索的关键词或术语", True),
                    ParamSpec("top_k", "integer", "返回片段数，默认 5", False),
                ],
                func=self.search_material,
            ),
            Tool(
                name="jump_chapter",
                description="跳转到指定章节并返回该章节完整正文，适合需要整章内容时使用。",
                parameters=[
                    ParamSpec("section", "string", "章节标题或章节路径片段", True),
                ],
                func=self.jump_chapter,
            ),
        ]

    def tool_defs(self) -> List[ToolDef]:
        return [
            ToolDef(t.name, t.description, t.parameters) for t in self.tools()
        ]

    def run(self, name: str, **args) -> str:
        for t in self.tools():
            if t.name == name:
                try:
                    return t.func(**args)
                except Exception as ex:  # noqa: BLE001 - 工具失败回填错误信息
                    return f"（工具 {name} 执行出错：{ex}）"
        return f"（未知工具：{name}）"