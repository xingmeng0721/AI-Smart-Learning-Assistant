"""DOCX 文档解析器：按 Word 内置 Heading 样式还原章节层级，产出 ParsedBlock。"""

from pathlib import Path
from typing import List

from docx import Document

from ingestion.schemas import ParsedBlock


class DocxLoader:
    """按 Word Heading/标题段落切分章节，正文段落归入当前章节。"""

    def load(self, path: str | Path, source: str = "") -> List[ParsedBlock]:
        path = Path(path)
        source = source or path.name
        doc = Document(str(path))
        stack: List[tuple[int, str]] = []
        buf: List[str] = []
        cur_level = 0
        cur_title = ""
        cur_path = ""
        blocks: List[ParsedBlock] = []

        def flush():
            nonlocal buf
            content = "\n".join(buf).strip()
            buf = []
            if not content:
                return
            blocks.append(
                ParsedBlock(
                    text=content,
                    title=cur_title,
                    section_path=cur_path,
                    level=cur_level,
                    source=source,
                )
            )

        for para in doc.paragraphs:
            level = self._heading_level(para)
            if level:
                flush()
                title = para.text.strip() or f"第{len(blocks)+1}节"
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                cur_level = level
                cur_title = title
                cur_path = " > ".join(t for _, t in stack)
            else:
                t = para.text.strip()
                if t:
                    buf.append(t)
        flush()
        return blocks

    def _heading_level(self, para) -> int:
        """根据段落样式名识别标题层级；返回 0 表示非标题。"""
        try:
            name = (para.style.name or "").lower()
        except Exception:
            return 0
        if name.startswith("heading"):
            num = name.replace("heading", "").strip()
            try:
                return int(num) if num else 1
            except ValueError:
                return 1
        # 中文「标题 1」等
        if "标题" in name:
            num = name.replace("标题", "").strip()
            try:
                return int(num) if num else 1
            except ValueError:
                return 1
        return 0