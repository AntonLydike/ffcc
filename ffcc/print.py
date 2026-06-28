import math
import sys
from itertools import count

from ffcc.ir import (
    IRNode,
    Kind,
    Value,
    MathNode,
    ConstantNode,
    VarNode,
    TunableNode,
    BitCastOperator,
    CastOperator,
    TestNode,
)
from io import TextIOBase, StringIO


def print_ssa(node: IRNode, file: TextIOBase = sys.stdout, **kwargs):
    # step 0: set up infra for naming values:
    names: dict[Value, str] = dict()
    used_names = set()
    printed = set()
    idx = count(0)

    def name(val: Value):
        # check if name hint is set
        if val.name is not None and not val.name[0].isnumeric():
            n = val.name
            i = 1
            while n in used_names:
                n = f"{val.name}{i}"
                i += 1
            names[val] = n
            used_names.add(n)
        # generate sequential name
        else:
            name = str(next(idx))
            names[val] = name
            used_names.add(name)

    # step 1: convert dag to list (in reverse dependency order)
    irbuff = []
    stack = [node]
    op: IRNode
    while stack:
        op = stack.pop()
        for arg in op.args:
            if arg.owner is None:
                name(arg)
                continue
            stack.append(arg.owner)

        irbuff.append(op)

    # step 2: iterate over the reversed list, and print items
    for op in reversed(irbuff):
        # print ops once
        if op in printed:
            continue
        printed.add(op)

        # assign names to results before printing
        # skip already named values
        if op.result in names:
            continue

        # assign name to result
        name(op.result)

        _print_ssa_node(op, names, file)


def _print_ssa_node(n: IRNode, names: dict[Value, str], out: TextIOBase):
    res = f"%{names[n.result]}"
    args = ", ".join(f"%{names[r]}" for r in n.args)
    match n:
        case MathNode(kind=Kind.Log, args=(arg, base), type=t):
            out.write(f"{res} = log %{names[arg]}, base=%{names[base]} : {t}\n")
        case MathNode(kind=k, type=t):
            out.write(f"{res} = {k.name.lower()} {args} : {t}\n")
        case ConstantNode(value=v, type=t):
            out.write(f"{res} = constant {v} : {t}\n")
        case VarNode(name=name, type=t):
            out.write(f"{res} = var {repr(name)} : {t}\n")
        case TunableNode(name=name, hint=h, type=t):
            out.write(f"{res} = tunable {repr(name)} = {h} : {t}\n")
        case BitCastOperator(direction, type=t, args=(a,)):
            out.write(f"{res} = bitcast {direction} {args} to {t}\n")
        case CastOperator(type=t, args=(a,)):
            out.write(f"{res} = cast {args} to {t}\n")
        case TestNode():
            out.write(f"{res} = test {args}\n")
        case _:
            print(type(n))
            print(n.args)
            print(n.result)
            raise ValueError(f"Unknown node", n)


def print_dag(node: IRNode, file: TextIOBase | None = None, **kwargs) -> str | None:
    out_was_none = file is None
    if out_was_none:
        file = StringIO()

    match node:
        case MathNode(kind=Kind.Log, argops=(arg, ConstantNode(base))) if base == int(
            base
        ):
            file.write(f"log{int(base)}(")
            print_dag(arg, file)
            file.write(")")
        case MathNode(kind=Kind.Log, argops=(arg, ConstantNode(math.e))):
            file.write("ln(")
            print_dag(arg, file)
            file.write(")")
        case MathNode(kind, argops):
            file.write(f"{kind.name.lower()}(")
            print_dag(argops[0], file)
            for op in argops[1:]:
                file.write(", ")
                print_dag(op, file)
            file.write(")")
        case ConstantNode(value):
            if value == math.e:
                file.write("e")
            elif value == math.pi:
                file.write("pi")
            else:
                file.write(f"{value}")
        case VarNode(name):
            file.write(f"{name}")
        case TunableNode(name=name, hint=h):
            file.write(f"tunable({repr(name)}={h})")
        case BitCastOperator(direction, argops=(op,)):
            file.write(f"{direction}(")
            print_dag(op, file)
            file.write(")")
        case CastOperator(argops=(op,), type=t):
            file.write(f"cast<{t}>(")
            print_dag(op, file)
            file.write(")")
        case TestNode(argops):
            file.write(f"test(")
            for op in argops:
                print_dag(op, file)
                file.write(", ")
            file.write(")")

    if out_was_none:
        return file.getvalue()
