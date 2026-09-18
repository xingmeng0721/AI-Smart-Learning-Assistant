"""文档解析器注册表：按文件后缀自动选择对应的 Loader。

- .md / .markdown / .txt → MarkdownLoader
- .pdf → PDFLoader（逐页 + 页码标记）
- .docx → DocxLoader（Word 内置标题层级）
"""

from pathlib import Path
from typing import List

from ingestion.loader.docx import DocxLoader
from ingestion.loader.markdown import MarkdownLoader
from ingestion.loader.pdf import PDFLoader
from ingestion.schemas import ParsedBlock

SUPPORTED_EXTS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def load_document(path: str | Path, source: str = "") -> List[ParsedBlock]:
    """读取文档并解析为 ParsedBlock 列表。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader().load(path, source=source)
    if suffix == ".docx":
        return DocxLoader().load(path, source=source)
    return MarkdownLoader().load(path, source=source)


__all__ = ["MarkdownLoader", "PDFLoader", "DocxLoader", "load_document", "SUPPORTED_EXTS"]