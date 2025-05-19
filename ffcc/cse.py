import logging

from ffcc.ir import IRNode
from ffcc.print import print_dag

LOGGER = logging.getLogger(__name__)


def structural_eq(n1: IRNode, n2: IRNode) -> bool:
    """
    check structural equivalency of two IR nodes.
    """
    if n1 is n2:
        return True
    if type(n1) is not type(n2):
        return False
    if len(n1.args) != len(n2.args):
        return False
    if not hasattr(type(n1), "val_attrs"):
        return False
    val_attrs = getattr(type(n1), "val_attrs")
    for attr in val_attrs:
        if getattr(n1, attr) != getattr(n2, attr):
            return False
    if all(structural_eq(a1, a2) for a1, a2 in zip(n1.argops, n2.argops, strict=True)):
        return True


def cse(node: IRNode) -> IRNode | None:
    # mark unique subexpressions we found
    unique_subexprs: list[IRNode] = []
    # elements to check
    worklist: list[IRNode] = [node]
    root = IRNode((node,))
    while worklist:
        for n in worklist.pop().args:
            op = n.owner
            for expr in unique_subexprs:
                # break if already scanned work is found
                if expr is op:
                    break
                # on structural equivalence, replace the current branch with the equivalent scanned one
                if structural_eq(expr, op):
                    n.replace_with(expr.result)
                    LOGGER.info(
                        f"Replacing {print_dag(op)} with equivalent result {print_dag(expr)}"
                    )
                    break
            # if the loop was not broken == if there was no equivalent object found
            else:
                # add this branch to the worklist
                worklist.append(op)
                unique_subexprs.append(op)

    return root.args[0].owner
