"""组件工厂：优先使用真实模型，不可用时自动降级本地实现。

优先级：
- Embedding: Ollama(bge-m3) → HashEmbedder
- Rerank:    本地 BgeReranker(bge-reranker-v2-m3) → Ollama → OverlapReranker
- LLM:       Ollama(qwen2.5) → MockLLM
"""

from pathlib import Path

import httpx

from config.settings import settings
from embedding.base import BaseEmbedder
from embedding.hash_embedder import HashEmbedder
from embedding.ollama_embedder import OllamaEmbedder
from llm.base import BaseLLM
from llm.mock_llm import MockLLM
from llm.ollama_llm import OllamaLLM
from retrieval.rerank.base import BaseReranker
from retrieval.rerank.bge_reranker import BgeReranker
from retrieval.rerank.ollama_reranker import OllamaReranker
from retrieval.rerank.overlap_reranker import OverlapReranker


def ollama_has_model(model: str) -> bool:
    """检查 Ollama 服务是否在线且已拉取指定模型。"""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return False
        names = {m.get("name", "") for m in r.json().get("models", [])}
        return any(n == model or n.startswith(model + ":") for n in names)
    except Exception:
        return False


def bge_reranker_available() -> bool:
    """本地 BgeReranker 模型目录是否就绪（含权重文件），不做重导入。"""
    p = Path(settings.bge_reranker_model)
    if not p.exists():
        return False
    return (
        (p / "model.safetensors").exists()
        or (p / "pytorch_model.bin").exists()
        or any(p.glob("*.safetensors"))
    )


def _reranker_fp16() -> bool:
    """返回 BgeReranker 是否启用 fp16（量化）。auto 表示有 CUDA 才开。"""
    v = settings.reranker_use_fp16.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    try:
        import torch  # noqa: WPS433

        return torch.cuda.is_available()
    except Exception:
        return False


def build_embedder() -> BaseEmbedder:
    if ollama_has_model(settings.embed_model):
        return OllamaEmbedder(
            settings.ollama_base_url,
            settings.embed_model,
            batch_size=settings.embed_batch_size,
        )
    return HashEmbedder(256)


def build_reranker() -> BaseReranker:
    # 1) 本地 bge-reranker-v2-m3 Cross-Encoder（最优先）
    if bge_reranker_available():
        return BgeReranker(
            model_path=settings.bge_reranker_model,
            top_k=settings.rerank_top_k,
            use_fp16=_reranker_fp16(),
            batch_size=settings.reranker_batch_size,
        )
    # 2) Ollama rerank（官方库暂无独立 rerank 模型，通常不可用）
    if ollama_has_model(settings.rerank_model):
        return OllamaReranker(
            settings.ollama_base_url,
            settings.rerank_model,
            top_k=settings.rerank_top_k,
        )
    # 3) 本地重叠兜底
    return OverlapReranker()


def build_llm() -> BaseLLM:
    if ollama_has_model(settings.llm_model):
        return OllamaLLM(
            settings.ollama_base_url,
            settings.llm_model,
            timeout=settings.ollama_timeout,
        )
    return MockLLM()


def components():
    """返回 (embedder, reranker, llm)，供 pipeline / 种库脚本共用。"""
    return build_embedder(), build_reranker(), build_llm()
