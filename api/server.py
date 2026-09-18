"""FastAPI 异步服务：流式问答、Agent 对话、知识库管理、健康检查、静态页面托管。"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.react_loop import ReactAgent
from agent.tools import AgentTools
from config.settings import settings
from ingestion.chunker.chunker import StructureChunker
from ingestion.loader import SUPPORTED_EXTS, load_document
from ingestion.loader.markdown import MarkdownLoader
from memory.conversation import ConversationMemory
from query.rewriter import QueryRewriter
from retrieval.hybrid import HybridRetriever
from retrieval.scoring import ConfidenceEvaluator
from service.components import build_embedder, build_llm, build_reranker, ollama_has_model
from service.rag import RagPipeline
from vectorstore.store import ChromaStore

app = FastAPI(title="AI 智能学习助教", version="0.1.0")

# 允许测试覆盖的存储目录（默认取 settings）
_chroma_dir_override: Optional[str] = None
# 允许测试覆盖的会话持久化路径（置空则纯内存）
_session_persist_override: Optional[str] = None

# 开发时前端（Vite dev server）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_pipeline():
    embedder = build_embedder()
    persist_dir = _chroma_dir_override or settings.chroma_persist_dir
    store = ChromaStore(persist_dir, embedder=embedder)
    retriever = HybridRetriever(store)
    retriever.sync_index()
    llm = build_llm()
    return RagPipeline(
        retriever=retriever,
        reranker=build_reranker(),
        evaluator=ConfidenceEvaluator(settings.confidence_threshold),
        llm=llm,
        memory=ConversationMemory(
            persist_path=_session_persist_override
            if _session_persist_override is not None
            else settings.session_persist_file
        ),
        rewriter=QueryRewriter(llm),
        rerank_top_k=settings.rerank_top_k,
        rerank_feed_top_k=settings.rerank_feed_top_k,
        rerank_enabled=settings.rerank_enabled,
        retrieval_top_k=settings.retrieval_top_k,
    )


# 模块级单例（便于复用）
_pipeline: Optional[RagPipeline] = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


@app.post("/agent/query")
async def agent_query(q: str = "", max_steps: int = 5):
    """Agent 问答：LLM 决策调用工具（笔记查询/资料检索/章节跳转）后给出答案。"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="q 不能为空")
    p = get_pipeline()
    agent = ReactAgent(AgentTools(p.retriever), llm=p.llm, max_steps=max_steps)
    return await agent.run(q)


@app.get("/health")
async def health():
    p = get_pipeline()
    mode = "ollama" if ollama_has_model(settings.embed_model) else "fallback"
    return {
        "status": "ok",
        "chunks": p.retriever._store.count(),
        "mode": mode,
        "llm": type(p.llm).__name__,
        "embedder": type(p.retriever._store._embedder).__name__,
        "reranker": type(p.reranker).__name__,
    }


@app.post("/query")
async def query(session_id: str = "default", q: str = ""):
    if not q.strip():
        raise HTTPException(status_code=400, detail="q 不能为空")

    async def event_gen():
        async for ev in get_pipeline().answer_stream(session_id, q):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest(request: Request, session_id: str = "", doc_id: str = "", text: str = ""):
    """增量导入：doc_id 已存在则先删除旧 Chunk 再写入（幂等）。
    支持查询参数或 JSON body（POST /ingest，body 含 doc_id/text）。"""
    if not text:
        try:
            body = await request.json()
            text = body.get("text", "")
            doc_id = doc_id or body.get("doc_id", "")
            session_id = session_id or body.get("session_id", "")
        except Exception:
            pass

    if not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id 不能为空")

    pipeline = get_pipeline()
    blocks = MarkdownLoader().parse(text, source=doc_id)
    return _store_blocks(pipeline, blocks, doc_id)


def _store_blocks(pipeline: RagPipeline, blocks, doc_id: str) -> dict:
    """分块、幂等覆盖写入向量库并重建 TF-IDF 索引。"""
    chunks = StructureChunker(settings.chunk_size).chunk_document(blocks, doc_id)
    pipeline.retriever._store.delete_doc(doc_id)  # 幂等：先清旧
    n = pipeline.retriever._store.upsert_docs(chunks)
    pipeline.retriever.sync_index()  # 增量更新 TF-IDF 索引
    return {"status": "ok", "doc_id": doc_id, "chunks": n, "total": pipeline.retriever._store.count()}


@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_id: str = ""):
    """上传文件导入知识库：按扩展名自动选择解析器（md/pdf/docx/txt）。"""
    name = file.filename or "document"
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {suffix or '(无后缀)'}，支持：{'、'.join(sorted(SUPPORTED_EXTS))}")

    data = await file.read()
    # 文本类直接解析，二进制类（pdf/docx）写临时文件解析
    did = doc_id or Path(name).stem
    if suffix in {".md", ".markdown", ".txt"}:
        text = data.decode("utf-8", errors="replace")
        blocks = MarkdownLoader().parse(text, source=name)
    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        try:
            blocks = load_document(tmp, source=name)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return _store_blocks(get_pipeline(), blocks, did)


@app.delete("/doc/{doc_id}")
async def delete_doc(doc_id: str):
    pipeline = get_pipeline()
    deleted = pipeline.retriever._store.delete_doc(doc_id)
    pipeline.retriever.sync_index()
    return {"status": "ok", "deleted": deleted}


@app.get("/documents")
async def list_docs():
    """文档列表：按 doc_id 聚合 chunk 数、标题、来源。"""
    p = get_pipeline()
    by_doc: Dict[str, dict] = {}
    for it in p.retriever._store.all_items():
        m = it.get("metadata", {}) or {}
        d = m.get("doc_id") or "?"
        rec = by_doc.setdefault(
            d,
            {
                "doc_id": d,
                "title": m.get("title") or d,
                "source": m.get("source") or "",
                "chunks": 0,
            },
        )
        rec["chunks"] += 1
    return {"docs": sorted(by_doc.values(), key=lambda x: x["doc_id"])}


@app.get("/doc/{doc_id}")
async def get_doc(doc_id: str):
    """查看某文档的 Chunk 列表（正文 + 章节路径）。"""
    p = get_pipeline()
    items = [
        it
        for it in p.retriever._store.all_items()
        if (it.get("metadata") or {}).get("doc_id") == doc_id
    ]
    chunks = []
    for it in items:
        m = it.get("metadata") or {}
        chunks.append(
            {
                "chunk_id": it["chunk_id"],
                "text": it["text"],
                "section_path": m.get("section_path") or "",
                "level": m.get("level"),
            }
        )
    if not chunks:
        raise HTTPException(status_code=404, detail="文档不存在或为空")
    return {"doc_id": doc_id, "total": len(chunks), "chunks": chunks}


@app.get("/sessions")
async def list_sessions():
    p = get_pipeline()
    return {"sessions": p.memory.list_sessions()}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    p = get_pipeline()
    return {"session_id": session_id, "messages": p.memory.get(session_id)}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    p = get_pipeline()
    p.memory.clear(session_id)
    return {"status": "ok", "session_id": session_id}


# 静态前端（Vue 构建产物）——路由需在 API 之后挂载
_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="web")
