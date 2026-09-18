"""会话记忆：基于内存维护多轮上下文，可选 JSON 文件持久化。"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ConversationMemory:
    """按 session_id 维护对话历史，支持追加、读取、清空、落盘持久化与按时间排序。

    内存约束：每个会话消息最多 max_turns*2 条；会话总数最多 max_sessions 条，
    超出后按最近活跃时间逐出最旧的（LRU），避免长期运行导致内存与落盘文件无限增长。
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_sessions: int = 200,
        persist_path: Optional[str] = None,
    ):
        self._store: Dict[str, List[dict]] = {}
        self._updated: Dict[str, float] = {}
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path is not None:
            self._load()
        self._evict_oldest(exclude=None)

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "sessions" in data and "updated" in data:
                    # 新格式：{sessions:{sid:msgs}, updated:{sid:ts}}
                    self._store = {
                        sid: msgs
                        for sid, msgs in data["sessions"].items()
                        if isinstance(msgs, list)
                    }
                    self._updated = {
                        sid: float(ts)
                        for sid, ts in data.get("updated", {}).items()
                        if isinstance(ts, (int, float))
                    }
                else:
                    # 旧格式：{sid: [msgs]}
                    self._store = {
                        sid: msgs for sid, msgs in data.items() if isinstance(msgs, list)
                    }
                    self._updated = {}
        except Exception:
            self._store = {}
            self._updated = {}

    def _save(self) -> None:
        if self.persist_path is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(
                {"sessions": self._store, "updated": self._updated},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _ts(epoch: Optional[float]) -> str:
        if not epoch:
            return ""
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")

    def add(self, session_id: str, role: str, content: str) -> None:
        self._store.setdefault(session_id, []).append(
            {"role": role, "content": content}
        )
        self._updated[session_id] = time.time()
        if len(self._store[session_id]) > self.max_turns * 2:
            self._store[session_id] = self._store[session_id][-self.max_turns * 2 :]
        self._evict_oldest(exclude=session_id)
        self._save()

    def _evict_oldest(self, exclude: Optional[str]) -> None:
        """超过 max_sessions 时，逐出最近活跃最旧的会话（LRU）。"""
        if self.max_sessions <= 0:
            self._store.clear()
            self._updated.clear()
            return
        while len(self._store) > self.max_sessions:
            # 找最旧（updated 最小）且非 exclude 的会话
            candidate = None
            cand_ts = float("inf")
            for sid, ts in self._updated.items():
                if sid == exclude:
                    continue
                if ts < cand_ts:
                    cand_ts = ts
                    candidate = sid
            if candidate is None:
                break
            self._store.pop(candidate, None)
            self._updated.pop(candidate, None)

    def get(self, session_id: str) -> List[dict]:
        return list(self._store.get(session_id, []))

    def context_for_llm(
        self,
        session_id: str,
        recent_messages: int = 8,
        anchor_queries: int = 6,
    ) -> List[dict]:
        """衰减式对话上下文（替代硬截断）。

        - 最近 recent_messages 条完整保留；
        - 更早的轮次不做硬删除，而是只保留其用户提问要点，压成一条
          "锚点" user 消息作为轻量记忆（token 随距离渐减 = 衰减）；
        - 无更早内容时直接返回最近消息。
        """
        msgs = self._store.get(session_id, [])
        if len(msgs) <= recent_messages:
            return list(msgs)
        recent = msgs[-recent_messages:]
        older = msgs[:-recent_messages]
        qs = [m["content"] for m in older if m["role"] == "user"]
        if not qs:
            return recent
        anchor = "（此前的提问要点）" + "；".join(qs[-anchor_queries:])
        return [{"role": "user", "content": anchor}] + recent

    def history_queries(self, session_id: str, limit: int = 4) -> List[str]:
        """取最近的用户提问（用于 Rewrite）。"""
        msgs = [m for m in self._store.get(session_id, []) if m["role"] == "user"]
        return [m["content"] for m in msgs[-limit:]]

    def list_sessions(self) -> List[dict]:
        """返回会话概要，按最近活跃时间降序。"""
        out = []
        for sid, msgs in self._store.items():
            if not msgs:
                continue
            out.append(
                {
                    "session_id": sid,
                    "turns": len(msgs) // 2,
                    "last": msgs[-1].get("content", ""),
                    "updated_at": self._ts(self._updated.get(sid)),
                }
            )
        return sorted(
            out,
            key=lambda x: self._updated.get(x["session_id"], 0.0),
            reverse=True,
        )

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._updated.pop(session_id, None)
        self._save()