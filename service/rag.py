"""RAG 问答编排：Query Rewrite → 混合检索 → Rerank → 置信度 → 生成。"""

from typing import AsyncIterator, Dict, List

from generation.generator import Generator, build_citations, build_citations_markdown, build_context
from llm.base import BaseLLM
from memory.conversation import ConversationMemory
from query.rewriter import QueryRewriter
from retrieval.hybrid import HybridRetriever
from retrieval.rerank.base import BaseReranker
from retrieval.scoring import ConfidenceEvaluator


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        evaluator: ConfidenceEvaluator,
        llm: BaseLLM,
        memory: ConversationMemory,
        rewriter: QueryRewriter,
        rerank_top_k: int = 5,
        rerank_feed_top_k: int = 8,
        rerank_enabled: bool = True,
        retrieval_top_k: int = 20,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.evaluator = evaluator
        self.llm = llm
        self.memory = memory
        self.rewriter = rewriter
        self.generator = Generator(llm)
        self.rerank_top_k = rerank_top_k
        self.rerank_feed_top_k = rerank_feed_top_k
        self.rerank_enabled = rerank_enabled
        self.retrieval_top_k = retrieval_top_k

    async def answer_stream(
        self, session_id: str, query: str
    ) -> AsyncIterator[Dict]:
        """流式问答，产出事件 dict：
        {'type':'rewrite','queries':[...]}
        {'type':'retrieval','count':n,'candidates':[...]}
        {'type':'refuse','reason':...} 或 {'type':'citations',...}
        {'type':'token','text':...}
        {'type':'done'}
        """
        history = self.memory.context_for_llm(session_id) if session_id else []
        rewritten = await self.rewriter.rewrite(query, self.memory.history_queries(session_id))
        yield {"type": "rewrite", "queries": rewritten}

        # 多路检索：对每个改写语句检索并合并
        all_cands: Dict[str, dict] = {}
        for q in rewritten:
            for c in self.retriever.retrieve(q, top_k=self.retrieval_top_k):
                all_cands.setdefault(c["chunk_id"], c)
        candidates = list(all_cands.values())
        # 按融合前分数降序，限制进入 Rerank 的候选数（缓解 Cross-Encoder CPU 开销）
        candidates.sort(
            key=lambda c: float(c.get("score", 0.0) or 0.0), reverse=True
        )
        feed = candidates[: self.rerank_feed_top_k]

        # Rerank 精排（可配置关闭，节省耗时）
        if self.rerank_enabled:
            ranked = self.reranker.rerank(query, feed)[: self.rerank_top_k]
        else:
            ranked = feed[: self.rerank_top_k]
        yield {"type": "retrieval", "count": len(ranked), "candidates": ranked}

        # 置信度 / 拒答
        verdict = self.evaluator.evaluate(ranked)
        if not verdict["in_kb"]:
            reason = "未在知识库中找到足够相关的资料，已拒绝回答。"
            yield {"type": "refuse", "reason": reason, "confidence": verdict["confidence"]}
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", reason)
            return

        context = build_context(ranked)
        citations = build_citations_markdown(ranked)
        yield {"type": "citations", "citations": build_citations(ranked)}

        # 流式生成
        full = []
        async for piece in self.generator.generate_stream(
            rewritten, context, citations, history
        ):
            full.append(piece)
            yield {"type": "token", "text": piece}
        self.memory.add(session_id, "user", query)
        self.memory.add(session_id, "assistant", "".join(full))
        yield {"type": "done"}
