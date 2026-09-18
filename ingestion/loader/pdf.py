"""PDF 文档解析器：逐页提取文本，为每个非空页生成一个 ParsedBlock 并标记页码。"""

from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

from ingestion.schemas import ParsedBlock


class PDFLoader:
    def load(self, path: str | Path, source: str = "") -> List[ParsedBlock]:
        path = Path(path)
        source = source or path.name
        reader = PdfReader(str(path))
        blocks: List[ParsedBlock] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            blocks.append(
                ParsedBlock(
                    text=text,
                    title=source,
                    section_path=source,
                    level=0,
                    page=i + 1,  # 页码从 1 开始，供引用溯源
                    source=source,
                )
            )
        return blocks