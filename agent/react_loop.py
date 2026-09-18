"""Agent ReAct 循环：LLM 决策 → 调用工具 → 观察回填 → 直至收敛。

仅依赖 BaseLLM 抽象与 AgentTools，便于测试注入 Mock/Fake LLM。
"""

import json
from typing import List

from agent.schemas import AgentDecision, AgentStep
from agent.tools import AgentTools
from llm.base import BaseLLM

_JSONDecoder = json.JSONDecoder()


class AgentReActError(Exception):
    """ReAct 运行异常（LLM 输出不可解析且无法继续）。"""


def _find_json_object(text: str):
    """返回文本中第一个完整可解析的 JSON 对象（支持嵌套），找不到返回 None。"""
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = _JSONDecoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    return None


def parse_decision(text: str) -> AgentDecision:
    """从 LLM 输出解析出 AgentDecision；优先取 JSON 对象，失败则尝试关键词兜底。"""
    payload = _find_json_object(text)
    if payload:
        kind = payload.get("action") or payload.get("kind")
        if kind in ("finish", "answer"):
            return AgentDecision("finish", answer=str(payload.get("answer", "")))
        if kind in ("tool", "call"):
            return AgentDecision(
                "tool",
                tool_name=str(payload.get("tool") or payload.get("tool_name", "")),
                tool_args={
                    str(k): str(v)
                    for k, v in dict(payload.get("tool_args") or payload.get("args") or {}).items()
                },
            )
    # 失败降级：无工具意图则视为问题无法推进
    if any(k in text for k in ("未能", "无法", "不调用", "没有", "未检索")):
        return AgentDecision("finish", answer=text)
    raise AgentReActError(f"无法解析 LLM 决策：{text[:200]}")


class ReactAgent:
    def __init__(
        self,
        tools: AgentTools,
        llm: BaseLLM,
        max_steps: int = 5,
    ):
        self._tools = tools
        self._llm = llm
        self.max_steps = max_steps

    def _system_prompt(self) -> str:
        defs = self._tools.tool_defs()
        lines = [
            "你是学习助教 Agent。请通过调用工具获取资料后回答问题。",
            "可用的工具（名称、说明、参数）：",
        ]
        for d in defs:
            params = ", ".join(
                f"{p.name}:{p.type}{'(必填)' if p.required else ''}" for p in d.parameters
            )
            lines.append(f"- {d.name}({params}): {d.description}")
        lines.append(
            "\n你必须严格以 JSON 输出决策，二选一：\n"
            '1. 调用工具：{"action":"tool","tool_name":"<名>","tool_args":{...}}\n'
            '2. 结束并回答：{"action":"finish","answer":"<最终答案，可标注 [来源N]>"}\n'
            "一次只输出一个 JSON 对象。"
        )
        return "\n".join(lines)

    async def run(self, query: str) -> dict:
        steps: List[AgentStep] = []
        scratchpad: List[str] = []
        user_msg = f"用户提问：{query}\n"

        for step_no in range(1, self.max_steps + 1):
            content = (
                self._system_prompt()
                + "\n\n"
                + user_msg
                + "推理历史：\n"
                + ("\n".join(scratchpad) if scratchpad else "（暂无）")
                + "\n\n请输出你的下一条决策："
            )
            llm_out = await self._llm.complete([{"role": "user", "content": content}])
            decision = parse_decision(llm_out)
            step = AgentStep(thought=llm_out.strip())

            if decision.is_finish:
                return {"answer": decision.answer, "steps": steps}

            # 调用工具
            observation = self._tools.run(decision.tool_name, **decision.tool_args)
            step.tool_name = decision.tool_name
            step.tool_args = decision.tool_args
            step.observation = observation
            steps.append(step)
            scratchpad.append(
                f"> 步骤{step_no} 调用 {decision.tool_name}({decision.tool_args})\n"
                f"观察：{observation[:500]}"
            )

        # 达到最大步数仍未 finish：兜底返回最后一个工具观察
        fallback = scratchpad[-1] if scratchpad else "（Agent 未能在步数内收敛）"
        return {"answer": fallback, "steps": steps, "truncated": True}