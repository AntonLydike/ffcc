import sys

from ffcc.ir import (
    IRNode,
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
    # step 1: convert dag to list (in reverse dependency order)
    irbuff = []
    stack = [node]
    idx = 0
    op: IRNode
    while stack:
        op = stack.pop()
        stack.extend((arg.owner for arg in op.args))
        irbuff.append(op)
    # step 2: iterate over the reversed list, and print items
    names: dict[Value, str] = dict()
    used_names = set()
    printed = set()
    for op in reversed(irbuff):
        # print ops once
        if op in printed:
            continue
        printed.add(op)

        # assign names to results before printing
        # skip already named values
        if op.result in names:
            continue
        # check if name hint is set
        if op.result.name is not None and not op.result.name[0].isnumeric():
            n = op.result.name
            i = 1
            while n in used_names:
                n = f"{op.result.name}{i}"
                i += 1
            names[op.result] = n
            used_names.add(n)
        # generate sequential name
        else:
            names[op.result] = idx
            used_names.add(idx)
            idx += 1
        _print_ssa_node(op, names, file)


def _print_ssa_node(n: IRNode, names: dict[Value, str], out: TextIOBase):
    res = f"%{names[n.result]}"
    args = ", ".join(f"%{names[r]}" for r in n.args)
    match n:
        case MathNode(kind=k, type=t):
            out.write(f"{res} = {k.name.lower()} {args} : {t}\n")
        case ConstantNode(value=v, type=t):
            out.write(f"{res} = constant {v} : {t}\n")
        case VarNode(name=n, type=t):
            out.write(f"{res} = var {repr(n)} : {t}\n")
        case TunableNode(name=n, hint=h, type=t):
            out.write(f"{res} = tunable {repr(n)} = {h} : {t}\n")
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
        case MathNode(kind, argops):
            file.write(f"{kind.name.lower()}(")
            print_dag(argops[0], file)
            for op in argops[1:]:
                file.write(", ")
                print_dag(op, file)
            file.write(")")
        case ConstantNode(value):
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
