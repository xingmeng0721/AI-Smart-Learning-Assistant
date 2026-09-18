"""TF-IDF 关键词检索索引：基于 scikit-learn TfidfVectorizer 对全库 Chunk 建立索引。"""

from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfIndex:
    """轻量倒排索引：fit 全量 Chunk 文本，query 返回按 TF-IDF 相似度排序的候选。"""

    def __init__(
        self,
        min_df: int = 1,
        char_range: tuple[int, int] | None = (2, 2),
    ):
        use_char = char_range is not None
        self._vectorizer = TfidfVectorizer(
            min_df=min_df,
            lowercase=True,
            analyzer="char_wb" if use_char else "word",
            ngram_range=char_range or (1, 1),
        )
        self._docs: List[str] = []  # chunk_id 列表，与矩阵行对应
        self._matrix = None
        self._fitted = False

    def rebuild(self, chunk_ids: List[str], texts: List[str]) -> None:
        """全量重建索引（增量场景下简单可靠）。"""
        self._docs = list(chunk_ids)
        if not texts:
            self._fitted = False
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def query(self, query: str, top_k: int = 10) -> List[dict]:
        """返回 [{'chunk_id','score','rank'}]，按相似度降序。"""
        if not self._fitted or not self._docs:
            return []
        qvec = self._vectorizer.transform([query])
        # 余弦相似度
        scores = (self._matrix @ qvec.T).toarray().ravel()
        order = scores.argsort()[::-1]
        out: List[dict] = []
        for rank, idx in enumerate(order[:top_k]):
            out.append(
                {
                    "chunk_id": self._docs[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return out
