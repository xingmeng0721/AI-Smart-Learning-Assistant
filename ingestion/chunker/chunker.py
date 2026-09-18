"""结构化分块器：基于 ParsedBlock 生成 Chunk，支持递归二分超长块。"""

from typing import Callable, List

from ingestion.schemas import Chunk, ParsedBlock

# token 估算函数（默认近似 4 字符/token，中文偏 1.5 字符/token，这里取保守值）
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 2)


class StructureChunker:
    """按章节层级分块；超长块按语义边界递归二分。"""

    def __init__(
        self,
        chunk_size: int = 512,
        token_counter: Callable[[str], int] = _approx_tokens,
    ):
        self.chunk_size = chunk_size
        self.token_counter = token_counter

    def chunk_document(
        self, blocks: List[ParsedBlock], doc_id: str
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        cid = 0
        for block in blocks:
            for part in self._split_block(block):
                chunk = Chunk(
                    text=part,
                    doc_id=doc_id,
                    title=block.title,
                    section_path=block.section_path,
                    level=block.level,
                    source=block.source,
                    page=block.page,
                    chunk_id=f"{doc_id}:{cid}",
                )
                chunks.append(chunk)
                cid += 1
        return chunks

    def _split_block(self, block: ParsedBlock) -> List[str]:
        text = block.text
        if self.token_counter(text) <= self.chunk_size:
            return [text]
        return self._recursive_split(text)

    def _recursive_split(self, text: str) -> List[str]:
        if self.token_counter(text) <= self.chunk_size:
            return [text]
        parts = self._split_at_boundary(text)
        if len(parts) <= 1:
            # 找不到边界，按字符硬切
            mid = len(text) // 2
            parts = [text[:mid], text[mid:]]
        result: List[str] = []
        for p in parts:
            result.extend(self._recursive_split(p))
        return result

    def _split_at_boundary(self, text: str) -> List[str]:
        """在段落/代码块边界处将文本一分为二（优先空行，其次代码块标记）。"""
        lines = text.splitlines(keepends=True)
        boundary = len(lines) // 2
        # 从中间向两侧找空行
        for i in range(boundary, len(lines)):
            if lines[i].strip() == "":
                return ["".join(lines[: i + 1]), "".join(lines[i + 1 :])]
        for i in range(boundary - 1, 0, -1):
            if lines[i].strip() == "":
                return ["".join(lines[: i + 1]), "".join(lines[i + 1 :])]
        return [text]
