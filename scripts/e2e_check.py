"""端到端验证脚本：调用本机 FastAPI 流式问答 + 增量导入 + 拒答。"""

import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"


def stream_query(session_id: str, q: str):
    print(f"\n=== query: {q} ===")
    with httpx.stream("POST", f"{BASE}/query", params={"session_id": session_id, "q": q}, timeout=30) as resp:
        resp.raise_for_status()
        tokens = []
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            ev = json.loads(line[len("data: "):])
            t = ev["type"]
            if t == "rewrite":
                print(f"  [rewrite] queries={ev['queries']}")
            elif t == "retrieval":
                print(f"  [retrieval] candidates={ev['count']}")
            elif t == "citations":
                for c in ev["citations"]:
                    print(f"    [cite] {c['section_path']} ({c['source']})")
            elif t == "refuse":
                print(f"  [refuse] {ev['reason']}")
            elif t == "token":
                tokens.append(ev["text"])
            elif t == "done":
                print(f"  [done] answer={''.join(tokens)}")


def main():
    # 1. 库内问答
    stream_query("s1", "什么是注意力机制")
    # 2. 多轮（第二次问可命中重写）
    stream_query("s1", "它有什么用")
    # 3. 库外拒答
    stream_query("s2", "火箭是怎么发射的")
    # 4. 增量导入 + 再查
    doc = "# 向量数据库\n\n向量数据库存储高维向量，支持相似度检索。"
    r = httpx.post(f"{BASE}/ingest", params={"doc_id": "new_doc", "text": doc}, timeout=30)
    print(f"\n[ingest] {r.json()}")
    stream_query("s3", "什么是向量数据库")


if __name__ == "__main__":
    main()
