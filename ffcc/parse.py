import re
from ast import literal_eval
from logging import getLogger

import math

LOGGER = getLogger(__name__)

from ffcc.ir import (
    IRNode,
    Kind,
    MathNode,
    Type,
    Value,
    FloatType,
    IntType,
    ConstantNode,
    VarNode,
    TunableNode,
    BitCastOperator,
    CastOperator,
    TestNode,
)

NAME_TO_KIND = {kind.name.lower(): kind for kind in Kind}


MATH_CONSTANTS = {
    'e': math.e,
    'pi': math.pi,
    'nan': math.nan,
    'inf': math.inf,
    '-inf': -math.inf,
}

def parse_ssa(text: str, baselineno: int = 0) -> IRNode:
    values: dict[str, Value] = dict()
    ops = []
    for lineno, line in enumerate(text.splitlines(), start=baselineno):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        parts = _split_line(line)
        val, eq = parts.pop(0), parts.pop(0)
        pos, valname = val
        if valname[0] != "%":
            raise ParseError(
                lineno, line, "Expected SSA result value here (%name)", pos
            )
        pos, eq = eq
        if eq != "=":
            raise ParseError(lineno, line, "Expected equals sign '=' here", pos)
        op: IRNode | None = None
        match parts:
            case [(_, "constant"), cst, (_, ":"), typ]:
                op = ConstantNode(
                    _parse_constant(lineno, line, *cst), _parse_type(lineno, line, *typ)
                )
            case [(_, "var"), name, (_, ":"), typ]:
                op = VarNode(
                    _parse_str_lit(lineno, line, *name), _parse_type(lineno, line, *typ)
                )
            case [(_, "tunable"), name, (_, "="), hint, (_, ":"), typ]:
                op = TunableNode(
                    _parse_str_lit(lineno, line, *name),
                    _parse_constant(lineno, line, *hint),
                    _parse_type(lineno, line, *typ),
                )
            case [(_, "bitcast"), direction, arg, (_, "to"), dst]:
                pos, dir = direction
                if dir not in ("i2f", "f2i"):
                    raise ParseError(
                        lineno, line, "Expected either f2i or i2f direction", pos
                    )
                op = BitCastOperator(
                    _parse_val(lineno, line, *arg, values),
                    dir,
                )
                res_t = _parse_type(lineno, line, *dst)
                if op.type != res_t:
                    raise ParseError(
                        lineno,
                        line,
                        f"Result type mismatch, expected {op.type}, got {res_t}",
                        pos,
                    )
            case [(_, "cast"), arg, (_, "to"), dst]:
                op = CastOperator(
                    _parse_val(lineno, line, *arg, values),
                    _parse_type(lineno, line, *dst),
                )
            case [(_, "test"), *args]:
                op = TestNode(
                    args=tuple(
                        _parse_val(lineno, line, *arg, values)
                        for arg in args
                        if arg[1] != ","
                    ),
                    result_types=(IntType(0),),
                )
            case [(pos, kind), lhs, (_, ","), rhs, (_, ":"), typ]:
                if kind not in NAME_TO_KIND:
                    raise ParseError(lineno, line, f"Unknown math operator {kind}", pos)
                op = MathNode(
                    _parse_val(lineno, line, *lhs, values),
                    _parse_val(lineno, line, *rhs, values),
                    kind=NAME_TO_KIND[kind],
                    res_type=_parse_type(lineno, line, *typ),
                )
            case [(pos, kind), arg, (_, ":"), typ]:
                if kind not in NAME_TO_KIND:
                    raise ParseError(lineno, line, f"Unknown math operator {kind}", pos)
                op = MathNode(
                    _parse_val(lineno, line, *arg, values),
                    kind=NAME_TO_KIND[kind],
                    res_type=_parse_type(lineno, line, *typ),
                )
            case [(pos, kind), *args, (_, ":"), typ]:
                if len(args) == 2:
                    raise ParseError(
                        lineno,
                        line,
                        f"Unknown operation format for {kind}: {args} (missing comma?)",
                        pos,
                    )
                raise ParseError(
                    lineno, line, f"Unknown operation format for {kind}", pos
                )
            case [(pos, kind), *args, (_, "to"), typ]:
                raise ParseError(
                    lineno,
                    line,
                    f"Unknown operation format for cast op {kind}: {args}",
                    pos,
                )
            case [(pos, kind), *args]:
                raise ParseError(
                    lineno,
                    line,
                    f"Unknown missing result type annotation for {kind}: {args}",
                    pos,
                )
        if op is None:
            raise ParseError(lineno, line, "Could not parse operation", 0)
        op.result.name = valname[1:]
        values[valname] = op.result
        ops.append((op, lineno, line))
    for op, lineno, line in ops[:-1]:
        if not op.result.uses:
            LOGGER.warning(f"Dead operation on line {lineno}: {line}")
    return ops[-1][0]


def _split_line(line: str) -> list[tuple[int, str]]:
    bits = []
    start = 0
    for i, c in enumerate(line):
        if c in "=():,":
            if i > start:
                bits.append((start, line[start:i]))
            bits.append((i, c))
            start = i + 1
        elif c == " ":
            if i > start:
                bits.append((start, line[start:i]))
            start = i + 1
    if start != len(line):
        bits.append((start, line[start:]))
    return bits


def _parse_val(
    lineno: int, line: str, pos: int, val: str, vals: dict[str, Value]
) -> Value:
    if "%" != val[0]:
        raise ParseError(lineno, line, "Expected SSA Value (%name)", pos)
    if val not in vals:
        raise ParseError(
            lineno, line, f"Value use before definition (falue name {repr(val)})", pos
        )
    return vals[val]


def _parse_constant(lineno: int, line: str, pos: int, text: str) -> int | float:
    if text in MATH_CONSTANTS:
        return MATH_CONSTANTS[text]
    cst = re.fullmatch(r"-?\d+(\.\d+)?([eE][+-]\d+)?", text)
    if cst is None:
        raise ParseError(lineno, line, "Expected int or float constant", pos)
    if "." in text or "e" in text.lower():
        val = float(text)
    else:
        val = int(text)
    return val


def _parse_type(lineno: int, line: str, pos: int, type: str) -> Type:
    if type[0] == "f":
        return FloatType(int(type[1:]))
    elif type[0] == "i":
        return IntType(int(type[1:]))
    raise ParseError(
        lineno,
        line,
        f"Malformed type, expected to start with f or i, got {repr(type)}",
        pos,
    )


def _parse_str_lit(lineno: int, line: str, pos: int, text: str) -> str:
    if text[0] not in ('"', "'"):
        text = f'"{text}"'
    try:
        return literal_eval(text)
    except (ValueError, TypeError, SyntaxError) as ex:
        raise ParseError(
            lineno, line, f"Malformed string literal {repr(text)}: {ex}", pos
        ) from ex
    except (MemoryError, RecursionError) as ex:
        raise ParseError(
            lineno, line, f"Serious error in string parsing: {ex}", pos
        ) from ex


class ParseError(BaseException):
    def __init__(self, lineno: int, line: str, message: str, offset: int):
        self.line = line
        self.message = message
        self.offset = offset
        self.lineno = lineno

    def __str__(self) -> str:
        pointer = "^"
        return f"ParseError in line {self.lineno}:\n{self.line}\n{pointer:>{self.offset+1}}\n{self.message}"
