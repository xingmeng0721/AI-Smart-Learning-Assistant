"""文档解析与分块的核心数据结构。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedBlock:
    """解析器产出的原始文本块，保留章节层级信息。"""

    text: str
    title: str = ""
    section_path: str = ""  # 形如 "1.2 引言 > 1.2.1 背景"
    level: int = 0
    page: Optional[int] = None
    source: str = ""  # 源文件名


@dataclass
class Chunk:
    """最终写入向量库的最小检索单元。"""

    text: str
    doc_id: str
    title: str
    section_path: str
    level: int
    source: str
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def to_metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "section_path": self.section_path,
            "level": str(self.level),
            "source": self.source,
            "page": str(self.page) if self.page is not None else "",
        }
