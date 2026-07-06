from dataclasses import Field, dataclass
import dataclasses
from functools import wraps
from typing import Any, Callable, Generator, Generic, TypeVar, cast, overload, Self

from ffcc.ir import IRNode, Value, IntType
from ffcc.print import print_dag

from logging import getLogger

LOGGER = getLogger(__name__)


@dataclass
class RewriteArgs:
    @classmethod
    def parse(cls, args_str) -> Self:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        args = {}
        tokens: list[str] = list(tokenize(args_str))

        def get_field(name: str) -> Field:
            if name in fields:
                return fields[name]
            raise ValueError(f"Unknown field: {name}, fields are {list(fields)}")

        while tokens:
            match tokens:
                case [name, "=", val, ",", *_] | [name, "=", val]:
                    field = get_field(name)
                    if "parser" in field.metadata:
                        args[name] = field.metadata["parser"](val)
                    elif field.type == "bool":
                        args[name] = val.lower() in ("1", "true", "y", "yes", "t")
                    else:
                        args[name] = field.type(val)  # pyright: ignore
                    tokens = tokens[4:]
                case [name, ",", *_] | [name]:
                    field = get_field(name)
                    if field.type is bool:
                        args[name] = True
                    else:
                        args[name] = field.type()  # pyright: ignore
                    tokens = tokens[2:]
                case [t, *_]:
                    raise ValueError(f"Unexpected token {repr(t)} in {repr(args_str)}")
        return cls(**args)


class RewriteResultModifiedOp:
    node: IRNode

    def __init__(self, node: IRNode | Value):
        self.node = node if isinstance(node, IRNode) else node.owner


def _add_arg(
    pattern: Callable[[IRNode], IRNode | RewriteResultModifiedOp | None],
) -> Callable[[IRNode, Any], IRNode | RewriteResultModifiedOp | None]:
    @wraps(pattern)
    def inner(node: IRNode, _: Any):
        return pattern(node)

    return inner


T = TypeVar("T", bound=RewriteArgs)


class Rewriter(Generic[T]):
    patterns: tuple[Callable[[IRNode, T], IRNode | RewriteResultModifiedOp | None], ...]
    args_t: type[T]

    @overload
    def __init__(
        self,
        patterns: tuple[
            Callable[[IRNode], IRNode | RewriteResultModifiedOp | None], ...
        ],
        args_t: None = None,
    ):
        pass

    @overload
    def __init__(
        self,
        patterns: tuple[
            Callable[[IRNode, T], IRNode | RewriteResultModifiedOp | None], ...
        ],
        args_t: type[T],
    ):
        pass

    def __init__(
        self,
        patterns: (
            tuple[Callable[[IRNode, T], IRNode | RewriteResultModifiedOp | None], ...]
            | tuple[Callable[[IRNode], IRNode | RewriteResultModifiedOp | None], ...]
        ),
        args_t: type[T] | None = None,
    ):
        if args_t is None:
            self.patterns = tuple(_add_arg(p) for p in patterns)
            self.args_t = RewriteArgs  # pyright: ignore
        else:
            self.patterns = cast(
                tuple[
                    Callable[[IRNode, T], IRNode | RewriteResultModifiedOp | None], ...
                ],
                patterns,
            )
            self.args_t = args_t

    def __call__(self, node: IRNode, conf: T) -> IRNode:
        return self.rewrite(node, conf)

    def rewrite(self, node: IRNode, conf: T) -> IRNode:
        seen = {node}
        worklist = [node]

        fake_root = IRNode((node,), IntType(0))
        del node

        while worklist:
            curr_node = worklist.pop()
            for pattern in self.patterns:
                new_node = pattern(curr_node, conf)
                if new_node is None or new_node is curr_node:
                    continue
                if isinstance(new_node, RewriteResultModifiedOp):
                    LOGGER.info(
                        f"applied {pattern.__name__} inplace: {print_dag(new_node.node)}"
                    )
                    worklist.append(new_node.node)
                    continue
                LOGGER.info(
                    f"applied {pattern.__name__}: {print_dag(curr_node)} -> {print_dag(new_node)}"
                )
                # add modified nodes to worklist
                for use in curr_node.result.uses:
                    if use in seen:
                        pass
                    worklist.append(use)
                    seen.add(use)

                curr_node.result.replace_with(new_node.result)
                worklist.append(new_node)
                break
            for arg in curr_node.args:
                if arg.owner in seen:
                    pass
                seen.add(arg.owner)
                worklist.append(arg.owner)

        # retrieve rewritten version of node
        return fake_root.args[0].owner

    def parse_args(self, args_str: str) -> T:
        return self.args_t.parse(args_str)

    def with_args(self, args: T) -> Callable[[IRNode], IRNode]:
        def inner(node: IRNode):
            return self(node, args)

        return inner


def tokenize(args_str: str) -> Generator[str, None, None]:
    end = len(args_str)
    p = start = 0
    while p < end:
        if args_str[p] in ",=":
            yield args_str[start:p]
            yield args_str[p]
            start = p = p + 1
        elif start == p and args_str[p] == '"':
            p += 1
            while args_str[p] != '"' and p < end:
                p += 1
            if args_str[p] != '"':
                raise ValueError(
                    f"Quote at {start} never closed! (remaining string: {repr(args_str[start:])})"
                )
            yield args_str[start + 1 : p]
            start = p = p + 1

        else:
            p += 1
    if p != start:
        yield args_str[start:p]
