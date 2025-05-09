from synth.ir import IRNode
from synth.printer import print_dag


def structural_eq(n1: IRNode, n2: IRNode) -> bool:
    """
    check structural equivalency of two IR nodes.
    """
    if type(n1) is not type(n2):
        return False
    if len(n1.args) != len(n2.args):
        return False
    if not all(structural_eq(a1, a1) for a1, a2 in zip(n1.argops, n2.argops)):
        return False
    if not hasattr(type(n1), 'val_attrs'):
        return False
    val_attrs = getattr(type(n1), 'val_attrs')
    for attr in val_attrs:
        if not getattr(n1, attr) == getattr(n2, attr):
            return False
    return True


def cse(node: IRNode) -> IRNode | None:
    subexprs: list[IRNode] = []
    worklist: list[IRNode] = [node]
    root = IRNode((node,))
    while worklist:
        for n in worklist.pop().args:
            for expr in subexprs:
                # break if already scanned work is found
                if expr is n.owner:
                    break
                # on structural equivalence, replace the equivalent branch with the deduplicated one
                if structural_eq(expr, n.owner):
                    n.replace_with(expr.result)
                    break
            # if the loop was not broken == if there was no equivalent object foun
            else:
                # add this branch to the worklist
                worklist.append(n.owner)
                subexprs.append(n.owner)

    return root.args[0].owner