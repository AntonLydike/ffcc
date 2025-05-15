import sys
from typing import Callable

from ffcc.ir import IRNode
from ffcc.printer import print_dag

from logging import getLogger

LOGGER = getLogger(__name__)


class Rewriter:
    patterns: tuple[Callable[[IRNode], IRNode | None], ...]

    def __init__(self, patterns: tuple[Callable[[IRNode], IRNode | None],...]):
        self.patterns = patterns

    def __call__(self, node: IRNode) -> IRNode:
        return self.rewrite(node)

    def rewrite(self, node: IRNode) -> IRNode:
        seen = {node}
        worklist = [node]

        fake_root = IRNode((node,))
        del node

        while worklist:
            curr_node = worklist.pop()
            for pattern in self.patterns:
                new_node = pattern(curr_node)
                if new_node is None or new_node is curr_node:
                    continue
                LOGGER.info(f"applied {pattern.__name__}: {print_dag(curr_node)} -> {print_dag(new_node)}")
                for old, new in zip(curr_node.results, new_node.results, strict=True):
                    # add modified nodes to worklist
                    for use in old.uses:
                        if use in seen:
                            pass
                        worklist.append(use)
                        seen.add(use)

                    old.replace_with(new)
                worklist.append(new_node)
                break
            for arg in curr_node.args:
                if arg.owner in seen:
                    pass
                seen.add(arg.owner)
                worklist.append(arg.owner)

        # retrieve rewritten version of node
        return fake_root.args[0].owner
