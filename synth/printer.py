from synth.ir import IRNode, Value, MathNode, ConstantNode, VarNode, TunableNode, BitCastOperator, CastOperator
from io import TextIOBase, StringIO


def print_ssa(node: IRNode, out: TextIOBase):
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
        for res in op.results:
            # skip already named values
            if res in names:
                continue
            # check if name hint is set
            if res.name is not None:
                n = res.name
                i = 1
                while n in used_names:
                    n = f'{res.name}{i}'
                    i += 1
                names[res] = n
                used_names.add(n)
            # generate sequential name
            else:
                names[res] = idx
                used_names.add(idx)
                idx += 1
        _print_ssa_node(op, names, out)

def _print_ssa_node(n: IRNode, names: dict[Value, str], out: TextIOBase):
    res = ', '.join(f'%{names[r]}' for r in n.results)
    args = ', '.join(f'%{names[r]}' for r in n.args)
    match n:
        case MathNode(kind=k, type=t):
            out.write(f'{res} = {k.name.lower()} {args} : {t}\n')
        case ConstantNode(value=v, type=t):
            out.write(f'{res} = constant {v} : {t}\n')
        case VarNode(name=n, type=t):
            out.write(f'{res} = var {repr(n)} : {t}\n')
        case TunableNode(name=n, hint=h, type=t):
            out.write(f'{res} = tunable {repr(n)} = {h} : {t}\n')
        case BitCastOperator(direction, type=t, args=(a,)):
            out.write(f'{res} = bitcast {direction} {args}  to {t}\n')
        case CastOperator(type=t, args=(a,)):
            out.write(f'{res} = cast {args} to {t}\n')
        case _:
            print(type(n))
            print(n.args)
            print(n.results)
            raise ValueError(f'Unknown node', n)


def print_dag(node: IRNode, out: TextIOBase | None = None) -> str | None:
    out_was_none = out is None
    if out_was_none:
        out = StringIO()

    match node:
        case MathNode(kind, argops):
            out.write(f'{kind.name.lower()}(')
            print_dag(argops[0], out)
            for op in argops[1:]:
                out.write(', ')
                print_dag(op, out)
            out.write(')')
        case ConstantNode(value):
            out.write(f'{value}')
        case VarNode(name):
            out.write(f'{name}')
        case TunableNode(name=name):
            out.write(f'tunable({repr(name)})')
        case BitCastOperator(direction, argops=(op,)):
            out.write(f'{direction}(')
            print_dag(op, out)
            out.write(')')
        case CastOperator(argops=(op,), type=t):
            out.write(f'cast<{t}>(')
            print_dag(op, out)
            out.write(')')

    if out_was_none:
        return out.getvalue()
