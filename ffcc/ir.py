from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, auto
from typing import Literal, Callable


class Kind(Enum):
    Add = auto()
    Mul = auto()
    Div = auto()
    Sub = auto()
    Pow = auto()
    Negate = auto()
    Log2 = auto()
    Floor = auto()

class Type:
    __match_args__ = ('width',)

    width: int

    def __init__(self, width: int):
        self.width = width

    def __eq__(self, other):
        return type(self) is type(other) and self.width == other.width

class IntType(Type):
    def __str__(self) -> str:
        return f'i{self.width}'

class FloatType(Type):
    def __str__(self) -> str:
        return f'f{self.width}'

class Value:
    __match_args__ = ('type', 'name')

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
            name = '0'
        return f'Value(%{name} : {self.type})'

    def __repr__(self):
        return f'{self.__class__.__name__}(type={self.type}, name={repr(self.name)}, owner={self.owner.__class__.__name__})'

    def replace_with(self, new_value: Value):
        for use in self.uses:
            use.args = tuple(
                val if val != self else new_value for val in use.args
            )
            new_value.uses.add(use)
        self.uses = set()


class IRNode:
    __match_args__ = ('argops', 'results', 'type')

    args: tuple[Value, ...]
    results: tuple[Value, ...]

    def __init__(
            self, 
            args: tuple[Value | IRNode, ...] = (),
            result_types: tuple[Type, ...] = (),
        ):
        self.args = tuple(arg.result if isinstance(arg, IRNode) else arg for arg in args)
        self.results = tuple(Value(typ, self) for typ in result_types)
        # add self as use
        for arg in self.args:
            arg.uses.add(self)

    @property
    def result(self) -> Value:
        assert len(self.results) == 1
        return self.results[0]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __str__(self):
        return f"{self.__class__.__name__}(args={self.args}, results={self.results})"

    @property
    def argops(self) -> tuple[IRNode, ...]:
        return tuple(val.owner for val in self.args)

    @property
    def type(self) -> Type:
        return self.result.type

    def walk(self, reverse:bool = False) -> Iterable[IRNode]:
        if not reverse:
            yield self

        for op in self.argops:
            yield from op.walk(reverse)

        if reverse:
            yield self


class MathNode(IRNode):
    __match_args__ = ('kind', 'argops', 'results', 'result', 'type')
    val_attrs = ('kind',)

    kind: Kind

    def __init__(self, *args: Value | IRNode, kind: Kind, res_type: Type):
        super().__init__(
            args=args,
            result_types=(res_type,)
        )
        self.kind = kind

    def __str__(self):
        return f"{self.__class__.__name__}<{self.kind.name}>(args={self.args}, results={self.results})"

class ConstantNode(IRNode):
    __match_args__ = ('value', 'results', 'type')
    val_attrs = ('value','type')

    value: int | float
    results: tuple[IntType | FloatType]

    def __init__(self, value: int | float, type: Type):
        super().__init__(
            result_types=(type,)
        )
        self.value = value

    def __str__(self):
        return f"{self.__class__.__name__}(value={self.value}, results={self.results})"

class VarNode(IRNode):
    __match_args__ = ('name', 'results', 'result', 'type')
    val_attrs = ('name', 'type')

    name: str

    def __init__(self, name: str, type: Type):
        super().__init__(
            result_types=(type,)
        )
        self.name = name
        self.result.name = name

    @property
    def type(self) -> Type:
        return self.result.type

    def __str__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, results={self.results})"

# Casting and Approximation Logic

class TunableNode(IRNode):
    __match_args__ = ('name', 'results', 'type', 'hint')

    name: str
    hint: float | int

    def __init__(self, name: str, hint: int | float, type: Type):
        super().__init__(
            result_types=(type,)
        )
        self.name = name
        self.hint = hint
        self.result.name = name

    @property
    def type(self) -> Type:
        return self.result.type

class BitCastOperator(IRNode):
    __match_args__ = ('direction', 'results', 'type')
    val_attrs = ('direction',)

    args: tuple[Value]
    direction: Literal['f2i', 'i2f']

    def __init__(self, value: Value | IRNode, direction: Literal['f2i', 'i2f']):
        if not isinstance(value, Value):
            value = value.result
        res_t = FloatType(value.type.width) if direction == 'i2f' else IntType(value.type.width)
        super().__init__(
            result_types=(res_t,),
            args=(value,)
        )
        self.direction = direction

class CastOperator(IRNode):
    __match_args__ = ('results', 'args', 'type')
    val_attrs = ('type',)

    args: tuple[Value]

    def __init__(self, value: Value | IRNode, type: Type):
        super().__init__(
            result_types=(type,),
            args=(value,)
        )

# testing only

class TestNode(IRNode):
    pass
