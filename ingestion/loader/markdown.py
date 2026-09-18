"""Markdown 文档解析器：按标题层级切分为结构化文本块。"""

import re
from pathlib import Path
from typing import List

from ingestion.schemas import ParsedBlock

# 匹配行首的 ATX 标题（# 到 ######）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownLoader:
    def load(self, path: str | Path, source: str = "") -> List[ParsedBlock]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.parse(text, source=source or path.name)

    def parse(self, text: str, source: str = "") -> List[ParsedBlock]:
        lines = text.splitlines()
        blocks: List[ParsedBlock] = []
        # 当前章节栈：[(level, title, path_part)]
        stack: List[tuple[int, str]] = []
        buf: List[str] = []
        cur_level = 0
        cur_title = ""
        cur_path = ""

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

        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                # 更新栈：弹出所有层级 >= 当前标题的
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                cur_level = level
                cur_title = title
                cur_path = " > ".join(t for _, t in stack)
            else:
                buf.append(line)
        flush()
        return blocks
