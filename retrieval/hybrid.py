"""混合检索器：向量检索 + TF-IDF 关键词检索双路召回，RRF 融合。"""

from typing import Dict, List, Optional

from retrieval.fusion import rrf_fusion
from retrieval.tfidf_index import TfidfIndex


class HybridRetriever:
    def __init__(self, vector_store, tfidf_index: Optional[TfidfIndex] = None):
        self._store = vector_store
        self._tfidf = tfidf_index or TfidfIndex()

    # ---------- 索引维护 ----------
    def sync_index(self) -> None:
        """从向量库全量拉取 chunk 文本，重建 TF-IDF 索引。"""
        items = self._store.all_items()
        chunk_ids = [i["chunk_id"] for i in items]
        texts = [i["text"] for i in items]
        self._tfidf.rebuild(chunk_ids, texts)

    # ---------- 双路召回 + 融合 ----------
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[Dict] = None,
        vector_weight: float = 1.0,
        tfidf_weight: float = 1.0,
    ) -> List[dict]:
        # 向量路
        vec_results = self._store.query(query, top_k=top_k, where=where)
        vec_ranked = [
            {"chunk_id": r["chunk_id"], "rank": i, "score": _cosine_score(r.get("distance"))}
            for i, r in enumerate(vec_results)
        ]
        # TF-IDF 路
        tf_ranked = self._tfidf.query(query, top_k=top_k)

        fused_ids = rrf_fusion(
            [vec_ranked, tf_ranked],
            weights=[vector_weight, tfidf_weight],
        )
        # 组装完整结果
        by_id = {r["chunk_id"]: r for r in vec_results}
        out: List[dict] = []
        for cid in fused_ids:
            base = by_id.get(cid, {})
            dist = base.get("distance")
            out.append(
                {
                    "chunk_id": cid,
                    "text": base.get("text", ""),
                    "metadata": base.get("metadata", {}),
                    "distance": dist,
                    "score": _cosine_score(dist),
                }
            )
        return out


def _cosine_score(distance) -> float:
    """Chroma 余弦距离转相似度分数。"""
    if distance is None:
        return 0.0
    return 1.0 - distance
