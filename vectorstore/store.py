"""ChromaDB 向量存储封装：写入、检索、增量更新、删除。"""

from typing import Dict, List, Optional

import chromadb

from ingestion.schemas import Chunk


class ChromaStore:
    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "documents",
        embedder=None,
    ):
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 写入 ----------
    def upsert_docs(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [c.chunk_id or f"{c.doc_id}:{i}" for i, c in enumerate(chunks)]
        texts = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]
        if self._embedder is not None:
            embeddings = self._embedder.embed(texts)
        else:
            embeddings = None  # Chroma 默认会自行嵌入（需配置）
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    # ---------- 检索 ----------
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> List[dict]:
        include = include or ["documents", "metadatas", "distances"]
        if self._embedder is not None:
            query_embeddings = self._embedder.embed([query_text])
            res = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=where,
                include=include,
            )
        else:
            res = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where,
                include=include,
            )
        out: List[dict] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for i, cid in enumerate(ids):
            out.append(
                {
                    "chunk_id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    # ---------- 增量更新 / 删除 ----------
    def delete_doc(self, doc_id: str) -> int:
        """删除某文档全部 Chunk，返回删除数量。"""
        res = self._collection.get(where={"doc_id": doc_id})
        ids = res.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def delete_all(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def all_items(self) -> List[dict]:
        """拉取全量 Chunk（用于重建 TF-IDF 索引等）。"""
        res = self._collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        out: List[dict] = []
        for i, cid in enumerate(ids):
            out.append(
                {
                    "chunk_id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return out
