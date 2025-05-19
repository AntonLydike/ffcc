from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from typing import Literal, ClassVar, Sequence
from ffcc.helper import CASTS
import copy

import math


class Kind(Enum):
    Add = auto()
    Mul = auto()
    Div = auto()
    Sub = auto()
    Pow = auto()
    Negate = auto()
    Log2 = auto()
    Floor = auto()
    Ashr = auto()  # arithmetic shift rigt
    Shl = auto()  # shift left


class Type:
    __match_args__ = ("width",)

    width: int

    def __init__(self, width: int):
        self.width = width

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

    def __init__(self, type: Type, owner: IRNode, name: str | None = None):
        self.name = name
        self.type = type
        self.owner = owner
        self.uses = set()

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
        for use in self.uses:
            use.args = tuple(val if val != self else new_value for val in use.args)
            new_value.uses.add(use)
        self.uses = set()


class IRNode:
    __match_args__ = ("argops", "args", "results", "type", "result")

    args: tuple[Value, ...]
    results: tuple[Value, ...]

    def __init__(
        self,
        args: tuple[Value | IRNode, ...] = (),
        result_types: tuple[Type, ...] = (),
    ):
        self.args = tuple(
            arg.result if isinstance(arg, IRNode) else arg for arg in args
        )
        self.results = tuple(Value(typ, self) for typ in result_types)
        # add self as use
        for arg in self.args:
            arg.uses.add(self)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return f"{self.__class__.__name__}(args={self.args}, results={self.results})"

    @property
    def result(self) -> Value:
        assert len(self.results) == 1
        return self.results[0]

    @property
    def argops(self) -> tuple[IRNode, ...]:
        return tuple(val.owner for val in self.args)

    @property
    def type(self) -> Type:
        return self.result.type

    def walk(self, reverse: bool = False) -> Iterable[IRNode]:
        if not reverse:
            yield self

        for op in self.argops:
            yield from op.walk(reverse)

        if reverse:
            yield self

    def __contains__(self, item: IRNode):
        if self is item:
            return True
        return any(item in op for op in self.argops)

    def copy(self) -> IRNode:
        return copy.deepcopy(self)

    def __add__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Add,
            res_type=self.type,
        )

    def __sub__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Sub,
            res_type=self.type,
        )

    def __mul__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Mul,
            res_type=self.type,
        )

    def __truediv__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Div,
            res_type=self.type,
        )

    def __pow__(self, power, modulo=None):
        assert modulo is None
        if not isinstance(power, IRNode):
            raise ValueError(power)
        return MathNode(
            self,
            power,
            kind=Kind.Pow,
            res_type=self.type,
        )

    def __neg__(self) -> IRNode:
        return MathNode(self, kind=Kind.Negate, res_type=self.type)

    def __lshift__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Shl,
            res_type=self.type,
        )

    def __rshift__(self, other: IRNode) -> IRNode:
        if not isinstance(other, IRNode):
            raise ValueError(other)
        return MathNode(
            self,
            other,
            kind=Kind.Ashr,
            res_type=self.type,
        )


class ConstantLikeNode(ABC, IRNode):
    __match_args__ = ("value", "argops", "args", "results", "type", "result")
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
    __match_args__ = ("argops", "args", "results", "type", "result", "evaluate")

    @abstractmethod
    def evaluate(self, args: list[float | int]) -> int | float | None:
        raise NotImplementedError()


class MathNode(FoldableNode):
    __match_args__ = ("kind", "argops", "results", "result", "type")
    val_attrs = ("kind",)

    kind: Kind

    def __init__(self, *args: Value | IRNode, kind: Kind, res_type: Type):
        super().__init__(args=args, result_types=(res_type,))
        self.kind = kind

    def __str__(self):
        return f"{self.__class__.__name__}<{self.kind.name}>(args={self.args}, results={self.results})"

    def evaluate(self, args: list[float | int]):
        match (self.kind, *args):
            case (Kind.Negate, a):
                return -a
            case (Kind.Log2, a):
                return math.log2(a)
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
            case (Kind.Shl, a, b):
                return a << b
            case (Kind.Ashr, a, b):
                return a >> b


class ConstantNode(ConstantLikeNode):
    __match_args__ = ("value", "results", "type")
    val_attrs = ("value", "type")

    value: int | float
    results: tuple[IntType | FloatType]

    def __init__(self, value: int | float, type: Type):
        super().__init__(result_types=(type,))
        self.value = value

    def with_new_value(
        self, new_val: int | float, replace_res_type: Type | None = None
    ) -> IRNode:
        if replace_res_type is None:
            replace_res_type = self.type
        return ConstantNode(new_val, replace_res_type)

    def __str__(self):
        return f"{self.__class__.__name__}(value={self.value}, results={self.results})"


class VarNode(IRNode):
    __match_args__ = ("name", "results", "result", "type")
    val_attrs = ("name", "type")

    name: str

    def __init__(self, name: str, type: Type):
        super().__init__(result_types=(type,))
        self.name = name
        self.result.name = name

    def __str__(self):
        return (
            f"{self.__class__.__name__}(name={repr(self.name)}, results={self.results})"
        )


# Casting and Approximation Logic


class TunableNode(ConstantLikeNode):
    __match_args__ = ("name", "results", "type", "hint", "value")

    priority: ClassVar[int] = 10

    name: str
    hint: float | int

    def __init__(self, name: str, hint: int | float, type: Type):
        super().__init__(result_types=(type,))
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


class BitCastOperator(FoldableNode):
    __match_args__ = ("direction", "results", "type", "argops", "result")
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
        super().__init__(result_types=(res_t,), args=(value,))
        self.direction = direction

    def evaluate(self, args: list[float | int]) -> int | float | None:
        (arg,) = args
        if (self.direction, self.args[0].type.width) in CASTS:
            return CASTS[self.direction, self.args[0].type.width](arg)


class CastOperator(FoldableNode):
    __match_args__ = ("results", "args", "type", "argops", "result")
    val_attrs = ("type",)

    args: tuple[Value]

    def __init__(self, value: Value | IRNode, type: Type):
        super().__init__(result_types=(type,), args=(value,))

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
