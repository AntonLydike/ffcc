from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import ctypes
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal, ClassVar, Sequence, Any, Self

from ffcc.helper import CASTS


class Kind(Enum):
    Add = auto()
    Mul = auto()
    Div = auto()
    Sub = auto()
    Pow = auto()
    Negate = auto()
    Log = auto()
    Floor = auto()
    Ashr = auto()  # arithmetic shift rigt
    Shl = auto()  # shift left


def _check_arg(arg: Any) -> IRNode | Value:
    if isinstance(arg, (IRNode, Value)):
        return arg
    elif isinstance(arg, int):
        return ConstantNode(arg, IntType(32))
    elif isinstance(arg, float):
        return ConstantNode(arg, FloatType(32))

    raise ValueError(f"Incompatible argument {arg} (of type {type(arg)})", arg)


@dataclass(frozen=True)
class Type:
    __match_args__ = ("width",)

    width: int

    def __eq__(self, other):
        return type(self) is type(other) and self.width == other.width

    @property
    @abstractmethod
    def ctype(self):
        raise NotImplementedError()


class IntType(Type):
    __match_args__ = ("width",)

    def __str__(self) -> str:
        return f"i{self.width}"

    @property
    def ctype(self):
        if self.width == 16:
            return ctypes.c_int16
        if self.width == 32:
            return ctypes.c_int32
        if self.width == 64:
            return ctypes.c_int64
        raise ValueError()


class FloatType(Type):
    __match_args__ = ("width",)

    def __str__(self) -> str:
        return f"f{self.width}"

    @property
    def ctype(self):
        if self.width == 32:
            return ctypes.c_float
        if self.width == 64:
            return ctypes.c_double
        raise ValueError()


class Value:
    __match_args__ = ("type", "name", "owner")

    type: Type
    name: str | None
    owner: IRNode
    uses: set[IRNode]
    is_frozen: bool = False

    def __init__(
        self, type: Type, owner: IRNode, name: str | None = None, frozen: bool = False
    ):
        self.name = name
        self.type = type
        self.owner = owner
        self.uses = set()
        self.is_frozen = frozen

    def __hash__(self):
        return id(self)

    def __eq__(self, value):
        return value is self

    def __str__(self) -> str:
        name = self.name
        if name is None:
            name = "0"
        return f"Value(%{name} : {self.type})"

    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.type}, name={repr(self.name)}, owner={self.owner.__class__.__name__})"

    def replace_with(self, new_value: Value):
        assert not self.is_frozen
        for use in self.uses:
            use.args = tuple(val if val != self else new_value for val in use.args)
            new_value.uses.add(use)
        self.uses = set()


class IRNode:
    __match_args__ = ("argops", "args", "result", "type")

    args: tuple[Value, ...]
    result: Value

    def __init__(
        self,
        args: tuple[Value | IRNode, ...] = (),
        result_type: Type = None,
    ):
        assert result_type is not None
        self.args = tuple(
            arg.result if isinstance(arg, IRNode) else arg for arg in args
        )
        self.result = Value(result_type, self)
        # add self as use
        for arg in self.args:
            if not arg.is_frozen:
                arg.uses.add(self)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        return f"{self.__class__.__name__}(args={self.args}, result={self.result})"

    def freeze(self) -> Self:
        """
        Make the IR immutable, results can no longer be re-written.
        :return:
        """
        self.result.is_frozen = True
        for arg in self.args:
            if not arg.is_frozen:
                arg.owner.freeze()
        return self

    @property
    def argops(self) -> tuple[IRNode, ...]:
        return tuple(val.owner for val in self.args)

    @property
    def type(self) -> Type:
        return self.result.type

    def inputs(self) -> list[VarNode]:
        res = []
        for node in self.walk():
            if isinstance(node, VarNode):
                res.append(node)
        return sorted(set(res), key=lambda n: res.index(n))

    def walk(self, reverse: bool = False) -> Iterable[IRNode]:
        if not reverse:
            yield self

        for op in self.argops:
            yield from op.walk(reverse)

        if reverse:
            yield self

    def subs(self, replace: dict[IRNode, IRNode]) -> IRNode:
        if self in replace:
            return replace[self]

        self.args = tuple(op.subs(replace).result for op in self.argops)

        return self

    def __contains__(self, item: IRNode):
        if self is item:
            return True
        return any(item in op for op in self.argops)

    def copy(self) -> IRNode:
        return copy.deepcopy(self)

    def __add__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Add,
            res_type=self.type,
        )

    def __sub__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Sub,
            res_type=self.type,
        )

    def __mul__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Mul,
            res_type=self.type,
        )

    def __truediv__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Div,
            res_type=self.type,
        )

    def __pow__(self, power: IRNode | Value | int | float, modulo=None):
        assert modulo is None
        return MathNode(
            self,
            _check_arg(power),
            kind=Kind.Pow,
            res_type=self.type,
        )

    def __neg__(self) -> IRNode:
        return MathNode(self, kind=Kind.Negate, res_type=self.type)

    def __lshift__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Shl,
            res_type=self.type,
        )

    def __rshift__(self, other: IRNode | Value | int | float) -> IRNode:
        return MathNode(
            self,
            _check_arg(other),
            kind=Kind.Ashr,
            res_type=self.type,
        )


class ConstantLikeNode(ABC, IRNode):
    __match_args__ = ("value", "argops", "args", "type", "result")
    priority: ClassVar[int] = 1

    value: int | float

    @abstractmethod
    def with_new_value(
        self, new_val: int | float, replace_res_type: Type | None = None
    ) -> IRNode:
        raise NotImplementedError()

    @staticmethod
    def make(
        new_val: int | float,
        replace_res_type: Type | None = None,
        from_ops: Sequence[ConstantLikeNode] = (),
    ) -> IRNode:
        new_op = max(from_ops, key=lambda op: op.priority)
        return new_op.with_new_value(new_val, replace_res_type)


class FoldableNode(ABC, IRNode):
    __match_args__ = ("argops", "args", "type", "result", "evaluate")

    @abstractmethod
    def evaluate(self, args: list[float | int]) -> int | float | None:
        raise NotImplementedError()


class MathNode(FoldableNode):
    __match_args__ = ("kind", "argops", "result", "type")
    val_attrs = ("kind",)

    kind: Kind

    def __init__(self, *args: Value | IRNode, kind: Kind, res_type: Type):
        super().__init__(args=args, result_type=res_type)
        self.kind = kind

    def __repr__(self):
        return f"{self.__class__.__name__}<{self.kind.name}>(args={self.args}, result={self.result})"

    def evaluate(self, args: Sequence[float | int]) -> float | int:
        match (self.kind, *args):
            case (Kind.Negate, a):
                return -a
            case (Kind.Log, a, b):
                return math.log(a, b)
            case (Kind.Floor, a):
                return math.floor(a)
            case (Kind.Pow, a, b):
                return a**b
            case (Kind.Add, a, b):
                return a + b
            case (Kind.Sub, a, b):
                return a - b
            case (Kind.Mul, a, b):
                return a * b
            case (Kind.Div, a, b):
                return a / b
            case (Kind.Shl, a, b) if isinstance(a, int) and isinstance(b, int):
                return a << b
            case (Kind.Ashr, a, b) if isinstance(a, int) and isinstance(b, int):
                return a >> b
            case other:
                raise RuntimeError("Unkown kind of math operation:", other)


class ConstantNode(ConstantLikeNode):
    __match_args__ = ("value", "result", "type")
    val_attrs = ("value", "type")

    value: int | float

    def __init__(self, value: int | float, type: Type, frozen: bool = False):
        super().__init__(result_type=type)
        self.value = value
        self.result.is_frozen = frozen

    def with_new_value(
        self, new_val: int | float, replace_res_type: Type | None = None
    ) -> IRNode:
        if replace_res_type is None:
            replace_res_type = self.type
        return ConstantNode(new_val, replace_res_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, result={self.result})"

    def __eq__(self, other):
        if isinstance(other, int | float):
            return self.value == other
        return self is other

    def __hash__(self):
        return id(self)


class VarNode(IRNode):
    __match_args__ = ("name", "result", "type")
    val_attrs = ("name", "type")

    name: str

    def __init__(self, name: str, type: Type):
        super().__init__(result_type=type)
        self.name = name
        self.result.name = name

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={repr(self.name)}, result={self.result})"
        )


# Casting and Approximation Logic


class TunableNode(ConstantLikeNode):
    __match_args__ = ("name", "result", "type", "hint", "value")

    priority: ClassVar[int] = 10

    name: str
    hint: float | int

    def __init__(self, name: str, hint: int | float, type: Type):
        super().__init__(result_type=type)
        self.name = name
        self.hint = hint
        self.result.name = name

    @property
    def value(self) -> int | float:
        return self.hint

    def with_new_value(
        self, new_val: int | float, replace_res_type: Type | None = None
    ) -> IRNode:
        if replace_res_type is None:
            replace_res_type = self.type
        return TunableNode(self.name, new_val, replace_res_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(args={self.args}, hint={repr(self.hint)}, result={self.result})"


class BitCastOperator(FoldableNode):
    __match_args__ = ("direction", "type", "argops", "result")
    val_attrs = ("direction",)

    args: tuple[Value]
    direction: Literal["f2i", "i2f"]

    def __init__(self, value: Value | IRNode, direction: Literal["f2i", "i2f"]):
        if not isinstance(value, Value):
            value = value.result
        res_t = (
            FloatType(value.type.width)
            if direction == "i2f"
            else IntType(value.type.width)
        )
        super().__init__(result_type=res_t, args=(value,))
        self.direction = direction

    def evaluate(self, args: list[float | int]) -> int | float | None:
        (arg,) = args
        if (self.direction, self.args[0].type.width) in CASTS:
            return CASTS[self.direction, self.args[0].type.width](arg)


class CastOperator(FoldableNode):
    __match_args__ = ("args", "type", "argops", "result")
    val_attrs = ("type",)

    args: tuple[Value]

    def __init__(self, value: Value | IRNode, type: Type):
        super().__init__(result_type=type, args=(value,))

    def evaluate(self, args: list[float | int]) -> int | float | None:
        (arg,) = args
        match self.type:
            case IntType():
                return int(arg)
            case FloatType():
                return float(arg)


# testing only


class TestNode(IRNode):
    pass
