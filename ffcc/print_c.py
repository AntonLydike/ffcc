from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence
from io import TextIOBase

from ffcc.ir import (
    BitCastOperator,
    CastOperator,
    ConstantLikeNode,
    FloatType,
    IntType,
    IRNode,
    Kind,
    MathNode,
    Type,
    Value,
    VarNode,
)
from ffcc.parse import Expression

KIND_TO_OP = {
    Kind.Add: "+",
    Kind.Mul: "*",
    Kind.Sub: "-",
    Kind.Div: "/",
    Kind.Shl: "<<",
    Kind.Ashr: ">>",
}

LOG_BASE_TO_OP = {
    2: "log2",
    10: "log10",
    math.e: "log",
}


def as_c_t(t: Type) -> str:
    match t:
        case IntType(width):
            if width in (16, 32, 64):
                return f"int{width}_t"
            raise ValueError(f"Unsupported int width: {width}")
        case FloatType(width):
            if width == 32:
                return "float"
            if width == 64:
                return "double"
            raise ValueError(f"Unsupported float width: {width}")


def _bitcast_helper(direction: str, width: int) -> tuple[str, str]:
    """
    Returns (function name, C code) for a bitcast of the given direction and
    bit width. Direction is "i2f" or "f2i".
    """
    if width == 32:
        ftype, itype = "float", "int32_t"
    elif width == 64:
        ftype, itype = "double", "int64_t"
    else:
        raise ValueError(f"Unsupported bitcast width: {width}")
    if direction == "i2f":
        name = f"ffcc_bc_int2{ftype}"
        code = f"""static inline {ftype} {name}({itype} x) {{
    union {{ {itype} i; {ftype} f; }} u;
    u.i = x;
    return u.f;
}}"""
    else:
        name = f"ffcc_bc_{ftype}2int"
        code = f"""static inline {itype} {name}({ftype} x) {{
    union {{ {itype} i; {ftype} f; }} u;
    u.f = x;
    return u.i;
}}"""
    return name, code


def _c_literal(value: int | float, t: Type) -> str:
    if isinstance(value, int):
        s = str(value)
        if isinstance(t, IntType) and t.width == 64 and abs(value) >= 2**31:
            s += "LL"
        return s
    if math.isnan(value):
        lit = "NAN"
    elif math.isinf(value):
        lit = "INFINITY" if value > 0 else "-INFINITY"
    else:
        lit = repr(value)
    if isinstance(t, FloatType) and t.width == 32:
        return f"{lit}f"
    return lit


def _c_func_name(name: str) -> str:
    """Convert a (possibly CamelCase) symbol name to a C function name."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def print_c(
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
        name = "FastFunction"
        vars = tuple(set(n for n in node.walk() if isinstance(n, VarNode)))
    if sym_name is not None:
        name = sym_name
    func_name = _c_func_name(name)

    # scan for required helpers and includes
    need_math_h = False
    need_stdint = False
    helpers: dict[str, str] = {}
    for op in node.walk():
        match op:
            case BitCastOperator(direction):
                fname, code = _bitcast_helper(direction, op.type.width)
                helpers[fname] = code
                need_stdint = True
            case MathNode(kind) if kind in (Kind.Log, Kind.Pow, Kind.Floor):
                need_math_h = True
            case ConstantLikeNode(value) if isinstance(value, float) and (
                math.isnan(value) or math.isinf(value)
            ):
                need_math_h = True
    # any integer-typed value (variables, results, casts, shifts) needs stdint.h
    if node.type is not None and isinstance(node.type, IntType):
        need_stdint = True
    for v in vars:
        if isinstance(v.type, IntType):
            need_stdint = True
    for op in node.walk():
        if isinstance(op, (CastOperator, MathNode)) and isinstance(op.type, IntType):
            need_stdint = True

    # write preamble
    if need_math_h:
        file.write("#include <math.h>\n")
    if need_stdint:
        if need_math_h and helpers:
            file.write("\n")
        file.write("#include <stdint.h>\n")
    for code in helpers.values():
        file.write(code + "\n")
    if need_stdint:
        file.write("\n")

    file.write(f"{as_c_t(node.type)} {func_name}(")
    file.write(", ".join(f"{as_c_t(v.type)} {v.name}" for v in vars))
    file.write(")\n{\n    ")

    lines: list[str] = []

    expr_to_str: dict[Value, str] = {v.result: v.name for v in vars}

    def make_var(val: Value) -> str:
        var = f"v{len(lines)}"
        lines.append(f"{as_c_t(val.type)} {var} = {expr_to_str[val]};")
        expr_to_str[val] = var
        return var

    def longer_arg_to_var(vals: Sequence[Value]) -> list[str]:
        longest = max(vals, key=lambda v: len(expr_to_str[v]))
        make_var(longest)
        return [expr_to_str[v] for v in vals]

    elem = node
    for elem in node.walk(reverse=True):
        if elem.result in expr_to_str:
            continue
        args = [expr_to_str[arg] for arg in elem.args]
        res = elem.result
        match elem:
            case ConstantLikeNode(value):
                expr_to_str[res] = _c_literal(value, res.type)
            case MathNode(kind=k) if k in KIND_TO_OP:
                op = KIND_TO_OP[k]
                if sum(map(len, args)) + 5 > line_width_limit:
                    args = longer_arg_to_var(elem.args)
                # cast args to the result type where required, so that the
                # operation is actually performed in the result type in C
                args = [
                    f"({as_c_t(res.type)})({a})" if arg.type != res.type else a
                    for a, arg in zip(args, elem.args)
                ]
                expr_to_str[res] = f"({args[0]} {op} {args[1]})"
            case MathNode(kind=Kind.Negate, args=(x,)):
                expr_to_str[res] = f"-{expr_to_str[x]}"
            case MathNode(kind=Kind.Log, argops=(x, ConstantLikeNode(base))):
                op = LOG_BASE_TO_OP.get(base)
                if op is None:
                    raise RuntimeError("Arbitrary-base logarithms not supported yet")
                if isinstance(res.type, FloatType) and res.type.width == 32:
                    op = f"{op}f"
                arg = expr_to_str[x.result]
                if len(arg) + len(op) + 2 > line_width_limit:
                    arg = make_var(x.result)
                expr_to_str[res] = f"{op}({arg})"
            case MathNode(kind=Kind.Pow):
                fn = (
                    "powf"
                    if isinstance(res.type, FloatType) and res.type.width == 32
                    else "pow"
                )
                if sum(map(len, args)) + len(fn) + 4 > line_width_limit:
                    args = longer_arg_to_var(elem.args)
                expr_to_str[res] = f"{fn}({', '.join(args)})"
            case MathNode(kind=Kind.Floor):
                fn = (
                    "floorf"
                    if isinstance(res.type, FloatType) and res.type.width == 32
                    else "floor"
                )
                expr_to_str[res] = f"{fn}({args[0]})"
            case BitCastOperator(direction, args=(x,)):
                fname, _ = _bitcast_helper(direction, res.type.width)
                arg = expr_to_str[x]
                if len(fname) + 2 + len(arg) > line_width_limit:
                    arg = make_var(x)
                expr_to_str[res] = f"{fname}({arg})"
            case CastOperator():
                arg = elem.args[0]
                expr_to_str[res] = f"({as_c_t(res.type)})({expr_to_str[arg]})"
        # make sure things that are used multiple times are always stored in variables
        if len(res.uses) > 1 and not isinstance(elem, ConstantLikeNode):
            make_var(res)

    lines.append(f"return {expr_to_str[elem.result]};")
    file.write("\n    ".join(lines) + "\n")
    file.write("}\n")
