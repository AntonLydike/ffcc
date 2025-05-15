from ffcc.ir import IRNode, TunableNode, ConstantNode
from ffcc.rewrite.rewriter import Rewriter


def instantiate(node: IRNode) -> IRNode:
    """
    Instantiate tuning nodes by their value
    """
    if isinstance(node, TunableNode):
        return ConstantNode(node.hint, node.type)


instantiate_pass = Rewriter((instantiate,))
