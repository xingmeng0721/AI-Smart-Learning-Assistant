"""示例知识库种库脚本：优先 Ollama Embedding，不可用时降级 HashEmbedder。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from ingestion.chunker.chunker import StructureChunker  # noqa: E402
from ingestion.loader.markdown import MarkdownLoader  # noqa: E402
from service.components import build_embedder  # noqa: E402
from vectorstore.store import ChromaStore  # noqa: E402

DEMO_DOCS = {
    "机器学习基础": """# 机器学习基础

## 1.1 什么是机器学习
机器学习是人工智能的一个分支，通过数据驱动的方式让模型自动学习规律。

## 1.2 监督学习
监督学习使用带标签的数据训练模型，常见任务包括分类与回归。

## 1.3 无监督学习
无监督学习在无标签数据上发现结构，常见任务包括聚类与降维。
""",
    "深度学习": """# 深度学习

## 2.1 神经网络
神经网络由多层神经元组成，通过反向传播更新参数。

## 2.2 注意力机制
注意力机制是 Transformer 的核心，允许模型关注输入序列不同位置。

## 2.3 自注意力
自注意力计算序列内部任意两个位置的相关性，是 Transformer 的基础组件。
""",
}


def main() -> None:
    embedder = build_embedder()
    store = ChromaStore(settings.chroma_persist_dir, embedder=embedder)
    chunker = StructureChunker(settings.chunk_size)
    loader = MarkdownLoader()

    total = 0
    for doc_id, md in DEMO_DOCS.items():
        store.delete_doc(doc_id)  # 幂等
        blocks = loader.parse(md, source=doc_id)
        chunks = chunker.chunk_document(blocks, doc_id)
        n = store.upsert_docs(chunks)
        total += n
        print(f"  [ingest] {doc_id}: {n} chunks")

    # 重建 TF-IDF 索引（由 API 启动时自动同步）
    print(f"完成：共 {total} 个 Chunk，向量库总数 {store.count()}。")


if __name__ == "__main__":
    main()
