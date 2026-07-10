from ctypes import cast
from dataclasses import dataclass
import re
from ast import Constant, TypeAlias, literal_eval
from logging import getLogger

import math
from typing import Literal, NoReturn


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
    "e": math.e,
    "pi": math.pi,
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
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
                    result_type=IntType(0),
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

            case [
                (pos, "log"),
                lhs,
                (_, ","),
                (_, "base"),
                (_, "="),
                base,
                (_, ":"),
                typ,
            ]:
                op = MathNode(
                    _parse_val(lineno, line, *lhs, values),
                    _parse_val(lineno, line, *base, values),
                    kind=Kind.Log,
                    res_type=_parse_type(lineno, line, *typ),
                )
            case [(pos, kind), arg, (_, ":"), typ] if (
                kind.startswith("log") or kind == "ln"
            ):
                try:
                    if kind == "ln":
                        base = math.e
                    else:
                        base = float(kind[3:])
                except ValueError as ex:
                    raise ParseError(
                        lineno, line, f"Malformed logarithm base: {kind[3:]}", pos + 3
                    ) from ex
                _t = _parse_type(lineno, line, *typ)
                op = MathNode(
                    _parse_val(lineno, line, *arg, values),
                    ConstantNode(base, _t),
                    kind=Kind.Log,
                    res_type=_t,
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
        return f"ParseError in line {self.lineno}:\n{self.line}\n{pointer:>{self.offset + 1}}\n{self.message}"


@dataclass
class Expression:
    name: str
    variables: tuple[VarNode, ...]
    expr: IRNode

    def __str__(self):
        from ffcc.print import print_dag
        from io import StringIO

        out = StringIO()
        print_dag(self.expr, out)

        return "Expression(name={}, variables=({}), expr={})".format(
            repr(self.name), ", ".join(v.name for v in self.variables), out.getvalue()
        )


def parse_expr(expr: str, float_t: FloatType = FloatType(32)) -> Expression:
    parts = re.fullmatch(
        r"([^(]+) *\(( *[a-zA-Z0-9]+(, *[A-Za-z0-9])*)\) *= (.*)", expr
    )
    if parts is None:
        raise ParseError(
            0,
            expr,
            "Malformed expression: Format must follow $name($vars) = $expression",
            0,
        )
    name = parts.group(1)
    vars = tuple(VarNode(name.strip(), float_t) for name in parts.group(2).split(","))
    vars_by_name = {v.name: v.result for v in vars}

    parsed_expr = ExpressionParser(parts.group(4), vars_by_name, float_t).parse().owner
    return Expression(name, vars, parsed_expr)


type TokenType = Literal[
    "NUMBER",
    "IDENTIFIER",
    "FUNCTION",
    "PLUS",
    "MINUS",
    "MUL",
    "DIV",
    "LPAREN",
    "RPAREN",
    "COMMA",
    "EOF",
]


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    start_pos: int  # Global character offset in the string


# Combined master regex for the unified parser state
TOKEN_REGEX = re.compile(
    r"(?P<NUMBER>\d+(?:\.\d+)?)"
    r"|(?P<FUNCTION>(?:pow|exp|ln|log\d+))(?=\s*\()"
    r"|(?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"|(?P<PLUS>\+)"
    r"|(?P<MINUS>-)"
    r"|(?P<MUL>\*)"
    r"|(?P<DIV>/)"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<SKIP>\s+)"
    r"|(?P<MISMATCH>.)"
)

# ==============================================================================
# 2. Parser Implementation
# ==============================================================================


class ExpressionParser:
    def __init__(
        self, expression: str, bound_vars: dict[str, Value], typ: FloatType
    ) -> None:
        self.expression = expression
        self.bound_vars = bound_vars
        self.length = len(expression)
        self.pos = 0
        self.typ = typ

        # Prime the first token
        self._current_token: Token = self._next_token()

    def _raise_error(
        self, message: str, position: int, ex: Exception | None = None
    ) -> NoReturn:
        """Helper to format and throw your exact ParseError structure."""
        # For a single expression string, we treat it as line 1.
        raise ParseError(
            lineno=1, line=self.expression, message=message, offset=position
        ) from ex

    def _next_token(self) -> Token:
        """Consumes internal string positions and returns the next valid Token."""
        while self.pos < self.length:
            match = TOKEN_REGEX.match(self.expression, self.pos)
            if not match:
                self._raise_error("Invalid syntax structure", self.pos)

            kind = match.lastgroup
            value = match.group()
            start = self.pos
            self.pos = match.end()  # Advance current pointer

            if kind == "SKIP":
                continue
            elif kind == "MISMATCH":
                self._raise_error(f"Unexpected character: '{value}'", start)
            elif kind is not None:
                return Token(kind, value, start)

        return Token("EOF", "", self.pos)

    def _consume(self, expected_type: TokenType | None = None) -> Token:
        tok = self._current_token
        if expected_type and tok.type != expected_type:
            self._raise_error(
                f"Expected token {expected_type}, got {tok.type}", tok.start_pos
            )

        # Advance token stream
        self._current_token = self._next_token()
        return tok

    def parse(self) -> Value:
        result = self._parse_expression()
        if self._current_token.type != "EOF":
            self._raise_error(
                "Unexpected data remaining at end of expression",
                self._current_token.start_pos,
            )
        return result

    def _parse_expression(self) -> Value:
        """Handles lowest precedence: + and -"""
        left_val = self._parse_term()

        while self._current_token.type in ("PLUS", "MINUS"):
            op_tok = self._consume()
            right_val = self._parse_term()
            left_val = MathNode(
                left_val,
                right_val,
                kind=Kind.Add if op_tok.value == "+" else Kind.Sub,
                res_type=self.typ,
            ).result

        return left_val

    def _parse_term(self) -> Value:
        """Handles medium precedence: * and /"""
        left_val = self._parse_factor()

        while self._current_token.type in ("MUL", "DIV"):
            op_tok = self._consume()
            right_val = self._parse_factor()
            left_val = MathNode(
                left_val,
                right_val,
                kind=Kind.Mul if op_tok.value == "*" else Kind.Div,
                res_type=self.typ,
            ).result

        return left_val

    def _parse_factor(self) -> Value:
        """Handles high precedence: Unary functions, Parentheses, Leaves"""
        tok = self._current_token

        match tok.type:
            case "FUNCTION":
                func_tok = self._consume()
                self._consume("LPAREN")
                arg_val = [self._parse_expression()]
                while self._current_token.type == "COMMA":
                    self._consume("COMMA")
                    arg_val.append(self._parse_expression())
                self._consume("RPAREN")

                if func_tok.value == "ln":
                    return MathNode(
                        *arg_val,
                        ConstantNode(math.e, self.typ),
                        kind=Kind.Log,
                        res_type=self.typ,
                    ).result
                if func_tok.value.startswith("log"):
                    try:
                        base = float(func_tok.value.removeprefix("log"))
                    except ValueError as ex:
                        self._raise_error(
                            f"Invalid log base: {func_tok.value.removeprefix('log')}",
                            func_tok.start_pos,
                            ex,
                        )
                    return MathNode(
                        *arg_val,
                        ConstantNode(base, self.typ),
                        kind=Kind.Log,
                        res_type=self.typ,
                    ).result
                if func_tok.value == "exp":
                    return MathNode(
                        ConstantNode(math.e, self.typ),
                        *arg_val,
                        kind=Kind.Pow,
                        res_type=self.typ,
                    ).result
                if func_tok.value.lower() not in NAME_TO_KIND:
                    self._raise_error(
                        f"Unknown math op: {func_tok.value}", func_tok.start_pos
                    )
                return MathNode(
                    *arg_val,
                    kind=NAME_TO_KIND[func_tok.value.lower()],
                    res_type=self.typ,
                ).result
            case "MINUS":
                self._consume("MINUS")
                val = self._parse_expression()
                return MathNode(val, kind=Kind.Negate, res_type=self.typ).result
            case "LPAREN":
                self._consume("LPAREN")
                val = self._parse_expression()
                self._consume("RPAREN")
                return val

            case "NUMBER":
                num_tok = self._consume("NUMBER")
                return ConstantNode(float(num_tok.value), self.typ).result
            case "IDENTIFIER":
                id_tok = self._consume("IDENTIFIER")
                var_name = id_tok.value
                if var_name not in self.bound_vars:
                    if var_name in MATH_CONSTANTS:
                        return ConstantNode(MATH_CONSTANTS.get(var_name), self.typ)
                    self._raise_error(
                        f"Undefined variable: '{var_name}'", id_tok.start_pos
                    )
                return self.bound_vars[var_name]

            case _:
                self._raise_error(
                    f"Unexpected symbol '{tok.value}' found while parsing factors",
                    tok.start_pos,
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    gr = parser.add_mutually_exclusive_group(required=True)
    gr.add_argument("-e", "--expression", help="Expression to parse")
    gr.add_argument("-i", "--input-file", help="Input file to read")

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Datatype to use throughout expression (default=f32)",
        default="f32",
    )

    args = parser.parse_args()
    if "expression" in args:
        typ = _parse_type(0, args.type, 0, args.type)
        expr = parse_expr(args.expression)
        print(expr)
    else:
        with open(args.input_file, "r") as f:
            ssa = parse_ssa(f.read())
            print(ssa)
