import ast
import math
import operator
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field

from app.schemas import StrictModel

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "在当前用户已授权的知识库检索事实依据",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算简单算术，禁止执行代码",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}


class SearchArguments(StrictModel):
    query: str = Field(min_length=1, max_length=2000)


class CalculatorArguments(StrictModel):
    expression: str = Field(min_length=1, max_length=200)


class ClockArguments(StrictModel):
    timezone: Literal["Asia/Shanghai", "UTC"] = "Asia/Shanghai"


CLOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "current_time",
        "description": "读取系统当前时间，仅限上海或 UTC 时区",
        "parameters": ClockArguments.model_json_schema(),
    },
}


def current_time(timezone: str = "Asia/Shanghai") -> str:
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 40:
        raise ValueError("计算式过长")
    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in unary:
            result = unary[type(node.op)](visit(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("指数过大")
            result = binary[type(node.op)](left, right)
        else:
            raise ValueError("只支持数字、括号及算术运算")
        if isinstance(result, complex) or not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("计算结果超出范围")
        return result

    return str(visit(tree))
