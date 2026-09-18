"""Agent 层数据结构：工具定义与 ReAct 决策。"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ParamSpec:
    name: str
    type: str = "string"  # string | integer
    description: str = ""
    required: bool = True


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: List[ParamSpec] = field(default_factory=list)


@dataclass
class ToolResult:
    """工具执行结果，回填到 ReAct scratchpad。"""

    tool_name: str
    ok: bool
    content: str


@dataclass
class AgentDecision:
    """LLM 单步决策：调用工具或结束并给出答案。"""

    kind: str  # "tool" | "finish"
    tool_name: str = ""
    tool_args: Dict[str, str] = field(default_factory=dict)
    answer: str = ""

    @property
    def is_finish(self) -> bool:
        return self.kind == "finish"


@dataclass
class AgentStep:
    """一次推理+动作的轨迹（用于可观测/测试）。"""

    thought: str = ""        # LLM 原始输出（决策 JSON）
    tool_name: str = ""
    tool_args: Dict[str, str] = field(default_factory=dict)
    observation: str = ""    # 工具返回