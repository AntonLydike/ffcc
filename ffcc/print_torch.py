from collections.abc import Sequence
from io import TextIOBase
from math import exp
import math
import sys

from ffcc.ir import (
    BitCastOperator,
    ConstantLikeNode,
    FloatType,
    IRNode,
    Kind,
    Type,
    Value,
    VarNode,
    MathNode,
    IntType,
)
from ffcc.parse import Expression

KIND_TO_OP = {
    Kind.Add: "+",
    Kind.Mul: "*",
    Kind.Sub: "-",
    Kind.Div: "/",
}


def quote_optional(text: str, node: IRNode) -> str:
    if isinstance(node, MathNode) and node.kind in (
        Kind.Add,
        Kind.Mul,
        Kind.Sub,
        Kind.Div,
        Kind.Negate,
    ):
        return f"({text})"
    return text


def as_torch_t(t: Type) -> str:
    if isinstance(t, IntType):
        return f"torch.int{t.width}"
    return f"torch.float{t.width}"


def print_torch(
    node: IRNode,
    file: TextIOBase = sys.stdout,
    expression: Expression | None = None,
    sym_name: str | None = None,
    line_width_limit: int = 40,
    **kwargs,
) -> None:
    if expression is not None:
        vars = expression.variables
        name = expression.name
    else:
        name = "FastModule"
        vars = tuple(set(n for n in node.walk() if isinstance(n, VarNode)))
    if sym_name is not None:
        name = sym_name

    # write preamble
    file.write(
        f"import torch\nfrom torch import nn, tensor\n\n\nclass {name}(nn.Module):\n"
    )
    file.write("\tdef forward(self, ")
    file.write(", ".join(f"{v.name} : tensor" for v in vars))
    file.write(") -> tensor:\n\t\t")

    lines: list[str] = []

    expr_to_str: dict[Value, str] = {v.result: v.name for v in vars}

    def make_var(val: Value) -> str:
        var = f"v{len(lines)}"
        lines.append(f"{var} = {expr_to_str[val]}")
        expr_to_str[val] = var
        return var

    def longer_arg_to_var(vals: Sequence[Value]) -> list[str]:
        longest = max(vals, key=lambda v: len(expr_to_str[v]))
        make_var(longest)
        return [expr_to_str[v] for v in vals]

    elem = node
    for elem in node.walk(reverse=True):
        args = [expr_to_str[arg] for arg in elem.args]
        res = elem.result
        match elem:
            case ConstantLikeNode(value):
                expr_to_str[res] = str(value)
            case MathNode(kind=k) if k in (Kind.Add, Kind.Sub, Kind.Div, Kind.Mul):
                op = KIND_TO_OP[k]
                if sum(map(len, args)) + 5 > line_width_limit:
                    args = longer_arg_to_var(elem.args)
                expr_to_str[res] = f"({args[0]} {op} {args[1]})"
            case MathNode(kind=Kind.Negate, args=(x,)):
                expr_to_str[res] = f"-{expr_to_str[x]}"
            case MathNode(kind=Kind.Log, argops=(x, ConstantLikeNode(val))):
                op = {
                    2: "log2",
                    10: "log10",
                    math.e: "log",
                }.get(val)
                if op is None:
                    raise RuntimeError("Arbitrary-base logarithms not supported yet")
                arg = expr_to_str[x.result]
                if len(arg) + len(op) + 2 > line_width_limit:
                    arg = make_var(x.result)
                expr_to_str[res] = f"{op}({arg})"
            case MathNode(kind=Kind.Pow):
                if sum(map(len, args)) + 11 > line_width_limit:
                    args = longer_arg_to_var(elem.args)
                expr_to_str[res] = f"torch.pow({', '.join(args)})"
            case BitCastOperator(direction, args=(x,)):
                op = f".view({as_torch_t(res.type)})"
                # insert cast if required
                if isinstance(x.type, IntType) and direction == "f2i":
                    op = f".type({as_torch_t(FloatType(x.type.width))}){op}"
                elif isinstance(x.type, FloatType) and direction == "i2f":
                    op = f".type({as_torch_t(IntType(x.type.width))}){op}"
                arg = expr_to_str[x]
                if len(op) + len(arg) > line_width_limit:
                    arg = make_var(x)
                expr_to_str[res] = f"{arg}{op}"

    lines.append(f"return {expr_to_str[elem.result]}\n")
    file.write("\n\t\t".join(lines))
