import sys
from typing import Callable

from ffcc.ir import IRNode, Value, IntType
from ffcc.print import print_dag

from logging import getLogger

LOGGER = getLogger(__name__)


class RewriteResultModifiedOp:
    node: IRNode

    def __init__(self, node: IRNode | Value):
        self.node = node if isinstance(node, IRNode) else node.owner


class Rewriter:
    patterns: tuple[Callable[[IRNode], IRNode | RewriteResultModifiedOp | None], ...]

    def __init__(self, patterns: tuple[Callable[[IRNode], IRNode | None], ...]):
        self.patterns = patterns

    def __call__(self, node: IRNode) -> IRNode:
        return self.rewrite(node)

    def rewrite(self, node: IRNode) -> IRNode:
        seen = {node}
        worklist = [node]

        fake_root = IRNode((node,), IntType(0))
        del node

        while worklist:
            curr_node = worklist.pop()
            for pattern in self.patterns:
                new_node = pattern(curr_node)
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
