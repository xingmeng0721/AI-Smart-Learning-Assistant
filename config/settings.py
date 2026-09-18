"""项目配置中心。"""

import os
from dataclasses import dataclass, field

# 项目根目录（config/ 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(key: str, default: str) -> str:
    """读取环境变量，未设置时返回默认值。"""
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Ollama 服务
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "bge-reranker-v2-m3"))
    # 本地 BgeReranker（BAAI/bge-reranker-v2-m3）模型目录（默认取项目内 models/，可用 BGE_RERANKER_MODEL 覆盖）
    bge_reranker_model: str = field(
        default_factory=lambda: _env(
            "BGE_RERANKER_MODEL", os.path.join(_PROJECT_ROOT, "models", "bge-reranker-v2-m3")
        )
    )

    # 分块
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 512))

    # 检索
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 20))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 5))
    # 进入 Rerank 打分前最多保留的候选数：限制 Cross-Encoder 计算量，缓解 BgeReranker CPU 开销
    rerank_feed_top_k: int = field(default_factory=lambda: _env_int("RERANK_FEED_TOP_K", 8))
    # 是否启用精排；追求速度时可置 false 走混合检索原始排序
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("RERANK_ENABLED", True))
    # BgeReranker 量化/批处理：fp16（GPU 下半精度量化加速，CPU 下关闭）、整批打分大小
    reranker_use_fp16: str = field(default_factory=lambda: _env("RERANKER_USE_FP16", "auto"))
    reranker_batch_size: int = field(default_factory=lambda: _env_int("RERANKER_BATCH_SIZE", 32))
    # OllamaEmbedder 分批向量化大小
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))
    # 置信度阈值：真实 Embedding（bge-m3）分数尺度更准确，默认 0.35；本地 HashEmbedder 尺度偏小，可放宽到 0.15
    confidence_threshold: float = field(default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.35))

    # 存储
    chroma_persist_dir: str = field(default_factory=lambda: _env("CHROMA_PERSIST_DIR", ".chroma"))
    # 会话持久化文件（JSON），置空即不落盘
    session_persist_file: str = field(default_factory=lambda: _env("SESSION_PERSIST_FILE", ".sessions/sessions.json"))

    # 请求超时（秒）
    ollama_timeout: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT", 120))


settings = Settings()
