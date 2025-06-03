"""
Do some synthesis for refinement kernels
"""
import itertools
import random
import sys
import time
from functools import reduce
from typing import Iterable, TypeVar
from collections.abc import Sequence, Iterator

import math
from aalib.multilines import MultilineCtx

from ffcc.helper import speedometer
from ffcc.print import print_ssa
from ffcc.ir import IRNode, VarNode, ConstantLikeNode, BitCastOperator, Kind, Value, ConstantNode, FloatType
from ffcc.eval import evaluate, partial_evaluation
import numpy as np

from aalib.progress import progress


f32 = FloatType(32)


def synthesize_refinement(approx: IRNode, exact: IRNode, domain: np.ndarray, epsilon: float = 0.0) -> IRNode:
    vars = approx.inputs()
    exact_vars = exact.inputs()

    if len(vars) == 1:
        approx_state =  {vars[0].result: domain}
        exact_args = {exact_vars[0].result: domain}
    else:
        approx_state = zip((v.result for v in vars), domain)
        exact_args = zip((v.result for v in exact_vars), domain)

    # get the exact results:
    exact_results = evaluate(exact, exact_args)

    # evaluate the approximation, but save intermediate values
    approx_baseline = evaluate(approx, approx_state)

    base_err = max_rel_err(approx_baseline, exact_results)


    # strip the suffix (ops that use a constant or input parameter)
    root = approx
    # make sure mutations to the IR are not tracked
    root.freeze()

    orig_root = root
    while len(root.args) == 2 and any(isinstance(arg.owner, (VarNode, ConstantLikeNode)) for arg in root.args):
        root = nonconst_arg(root)

    base_err = base_err
    current_p = None

    t0 = time.time()
    ctx = MultilineCtx(2)

    for i, p in enumerate(synthesize_with_vars(root, vars, 10, ctx.ostream_for(1))):
        res = partial_evaluation(p, approx_state)
        if orig_root is not root:
            res = partial_evaluation(orig_root, {**approx_state, root.result: res})

        if i != 0 and i % 10000 == 0:
            speedometer(t0, i, f'err={base_err}', file=ctx.ostream_for(0))

        err = max_rel_err(res, exact_results, epsilon)
        if err + 1e-6 < base_err:
            current_p = p
            current_err = err
            print(f"=========================\ncandidate: (err = {err} < {current_err})")
            print_ssa(p)

    return current_p

def max_rel_err(approx: np.ndarray, exact: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return np.max(np.abs(approx - exact) / (np.abs(exact) + eps))


def nonconst_arg(node: IRNode):
    for arg in node.args:
        if not isinstance(arg.owner, (VarNode, ConstantLikeNode)):
            return arg.owner
    raise ValueError("Node must have one non-const, non-var argument")



# we want to use:
#  - all variables
#  - prefer operands to bitcast ops (these are pre-error variables)
#  - simple constants
def synthesize_with_vars(root: IRNode, vars: list[VarNode], max_len: int = 10, file = sys.stdout) -> Iterable[IRNode]:
    values_to_use = [
        *vars,
        *(op.args[0].owner for op in root.walk() if isinstance(op, BitCastOperator))
    ]

    constants = []
    counts = len(values_to_use)
    g = constant_generator()

    for d in range(2, max_len):
        constants.append(next(g))
        counts += 1

        for vars in progress(itertools.combinations((*values_to_use, *constants), d-1), file=file, message=f"depth={d}", count=math.comb(counts, d-1)):
            yield from programs_containing(root, list(vars))


def programs_containing(base: IRNode, vars: list[IRNode]) -> Iterator[IRNode]:
    if not vars:
        yield base

    for i, var in enumerate(vars):
        rest = vars[:i] + vars[i+1:]

        yield from programs_containing(var - base, rest)
        yield from programs_containing(base + var, rest)
        yield from programs_containing(base - var, rest)

        if var != 1:
            yield from programs_containing(base * var, rest)




def constant_generator() -> Iterator[ConstantNode]:
    yield ConstantNode(1, f32, frozen=True)
    n = 1
    while True:
        n += 1
        yield ConstantNode(n, f32, frozen=True)
        yield ConstantNode(1/n, f32, frozen=True)
        for i in range(2, n):
            if i < n:
                yield ConstantNode(i/n, f32, frozen=True)


T = TypeVar('T')
def prod(__iterable: Iterable[T], start: T = 1):
    for e in __iterable:
        start *= e
    return start



if __name__ == '__main__':
    from ffcc.parse import parse_ssa
    orig = parse_ssa("""%x = var x : f32
    %mh = constant -0.5 : f32
    %r = pow %x, %mh : f32
    """)

    approx = parse_ssa("""%sigma = tunable 'sigma' = 1597463007 : i32
    %x = var 'x' : f32
    %0 = bitcast f2i %x to i32
    %1 = constant 1 : i32
    %2 = ashr %0, %1 : i32
    %3 = negate %2 : i32
    %4 = add %sigma, %3 : i32
    %5 = bitcast i2f %4 to f32""")

    synthesize_refinement(approx, orig, np.linspace(1, 4, dtype=np.float32))
