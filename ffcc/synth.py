"""
Do some synthesis for refinement kernels
"""

import itertools
import sys
import time
from typing import Iterable, TypeVar
from collections.abc import Sequence, Iterator

from aalib.multilines import MultilineCtx

from ffcc.helper import speedometer
from ffcc.print import print_ssa
from ffcc.ir import (
    IRNode,
    VarNode,
    ConstantLikeNode,
    BitCastOperator,
    Kind,
    Value,
    ConstantNode,
    FloatType,
    MathNode,
)
from ffcc.eval import evaluate, partial_evaluation
import numpy as np

from aalib.progress import progress


f32 = FloatType(32)


def synthesize_refinement(
    approx: IRNode,
    exact: IRNode,
    domain: np.ndarray,
    epsilon: float = 0.0,
    timeout: float = 6 * 60,
    err_cutoff=0.0,
) -> IRNode:
    vars = approx.inputs()
    exact_vars = exact.inputs()

    if len(vars) == 1:
        approx_state = {vars[0].result: domain}
        exact_args = {exact_vars[0].result: domain}
    else:
        approx_state = zip((v.result for v in vars), domain)
        exact_args = zip((v.result for v in exact_vars), domain)

    # get the exact results:
    exact_results = evaluate(exact, exact_args)

    # evaluate the approximation, but save intermediate values
    approx_baseline = evaluate(approx, approx_state)

    base_err = max_rel_err(approx_baseline, exact_results, epsilon)

    # strip the suffix (ops that use a constant or input parameter)
    root = approx
    # make sure mutations to the IR are not tracked
    root.freeze()

    orig_root = root
    while len(root.args) == 2 and any(
        isinstance(arg.owner, (VarNode, ConstantLikeNode)) for arg in root.args
    ):
        root = nonconst_arg(root)

    t0 = time.time()
    ctx = MultilineCtx(3)

    current_p = None
    current_err = base_err
    current_subs = {}

    try:
        for i, (p, assignment) in enumerate(
            synthesize_with_vars(root, vars, 10, ctx.ostream_for(1), ctx.ostream_for(2))
        ):
            res = partial_evaluation(
                p,
                {
                    h: approx_state[v] if v in approx_state else v.owner.value
                    for h, v in assignment.items()
                },
            )

            if orig_root is not root:
                approx_state[root.result] = res
                res = partial_evaluation(orig_root, approx_state)

            if i != 0 and i % 10000 == 0:
                speedometer(t0, i, f"err={base_err}", file=ctx.ostream_for(0))
                if time.time() - t0 > timeout:
                    print("timeout hit")
                    break

            err = max_rel_err(res, exact_results, epsilon)
            if err + 1e-6 < current_err:
                print(
                    f"\n\n\n=========================\ncandidate: (err = {err} < {current_err}, improvement = {current_err - err})"
                )
                current_p = p
                current_err = err
                current_subs = assignment
                print_ssa(current_p)
                print("\n")
                if current_err < err_cutoff:
                    print("accuracy hit")
                    break
    except KeyboardInterrupt:
        if current_p is None:
            print("No valid program found, raising exception again")
            raise

    current_p = apply_subs(current_p, current_subs)

    if orig_root is not root:
        return apply_subs(orig_root, {root.result: current_p.result})

    return current_p


def max_rel_err(approx: np.ndarray, exact: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return np.max(np.abs(approx - exact) / (np.abs(exact) + eps))


def nonconst_arg(node: IRNode):
    for arg in node.args:
        if not isinstance(arg.owner, (VarNode, ConstantLikeNode)):
            return arg.owner
    raise ValueError("Node must have one non-const, non-var argument")


def synthesize_with_vars(
    root: IRNode,
    variables: Sequence[VarNode],
    max_len: int = 10,
    generation_progress=sys.stdout,
    fine_progress=sys.stdout,
) -> Iterable[tuple[IRNode, dict[Value, Value]]]:
    values_to_use: list[Value] = [
        *(op.result for op in variables),
        *(op.args[0] for op in root.walk() if isinstance(op, BitCastOperator)),
    ]

    start_depth = 3

    g = constant_generator()

    constants: list[Value] = [next(g).result]
    counts = len(values_to_use) + len(constants)
    holes: list[Value] = [
        Value(f32, None, frozen=True, name="hole") for _ in range(start_depth - 1)
    ]

    base_var = root.result

    for d in range(start_depth, max_len):
        constants.append(next(g).result)
        holes.append(Value(f32, None, frozen=True, name="hole"))
        counts += 1
        # len(values_to_use) + len(constants) = counts
        # len(holes) == d
        assert len(holes) == d
        # print(f"depth = {d}, variables = {counts}", file=generation_progress)

        i = 0
        for i, sketch in progress(
            list(enumerate(program_sketches(holes))),
            message=f"depth={d}",
            file=fine_progress,
        ):
            for combination in itertools.product(
                (base_var, *values_to_use, *constants), repeat=d
            ):
                if base_var not in combination:
                    continue
                # return a dict mapping hole -> real value
                yield sketch, dict(zip(holes, combination, strict=True))
        print(f"{i} sketches at depth {d}", file=generation_progress)


def program_sketches(holes: list[Value]):
    # smallest sketch has three holes
    assert len(holes) >= 3
    # case with three nodes: yield (a * b) + c, (a * b) - c, (a * b) * c
    if len(holes) == 3:
        a, b, c = holes
        mul = MathNode(a, b, kind=Kind.Mul, res_type=f32).freeze()
        yield MathNode(mul, c, kind=Kind.Add, res_type=f32).freeze()
        yield MathNode(mul, c, kind=Kind.Sub, res_type=f32).freeze()
        yield MathNode(c, mul, kind=Kind.Sub, res_type=f32).freeze()
        yield MathNode(mul, c, kind=Kind.Mul, res_type=f32).freeze()
        return

    hole = holes[1]
    # for more kinds, first pop the hole on the outside, yielding x * [programs], x + [programs] and x - [programs]
    for tree in program_sketches(holes[1:]):
        yield MathNode(hole, tree, kind=Kind.Mul, res_type=f32).freeze()
        yield MathNode(hole, tree, kind=Kind.Add, res_type=f32).freeze()
        yield MathNode(hole, tree, kind=Kind.Sub, res_type=f32).freeze()

    # then pop the hole on the inside:
    a, b, *holes = holes
    for new_hole in (
        MathNode(a, b, kind=Kind.Mul, res_type=f32).freeze(),
        MathNode(a, b, kind=Kind.Sub, res_type=f32).freeze(),
        MathNode(a, b, kind=Kind.Add, res_type=f32).freeze(),
    ):
        yield from program_sketches([new_hole.result, *holes])


def constant_generator() -> Iterator[ConstantNode]:
    yield ConstantNode(1, f32, frozen=True)
    n = 1
    while True:
        n += 1
        yield ConstantNode(n, f32, frozen=True)
        yield ConstantNode(1 / n, f32, frozen=True)
        for i in range(2, n):
            if i < n:
                yield ConstantNode(i / n, f32, frozen=True)


T = TypeVar("T")


def prod(__iterable: Iterable[T], start: T = 1):
    for e in __iterable:
        start *= e
    return start


def apply_subs(p: IRNode, subs: dict[Value, Value]):
    new_args = []
    for arg in p.args:
        if arg in subs:
            new_args.append(subs[arg])
        else:
            new_args.append(arg)
            if arg.owner is not None:
                apply_subs(arg.owner, subs)

    p.args = tuple(new_args)
    return p


if __name__ == "__main__":
    from ffcc.parse import parse_ssa

    orig = parse_ssa(
        """%x = var x : f32
    %mh = constant -0.5 : f32
    %r = pow %x, %mh : f32
    """
    )

    approx = parse_ssa(
        """%sigma = tunable 'sigma' = 1597463007 : i32
    %x = var 'x' : f32
    %0 = bitcast f2i %x to i32
    %1 = constant 1 : i32
    %2 = ashr %0, %1 : i32
    %3 = negate %2 : i32
    %4 = add %sigma, %3 : i32
    %5 = bitcast i2f %4 to f32"""
    )

    result = synthesize_refinement(
        approx, orig, np.linspace(1, 4, num=100, dtype=np.float32), epsilon=0.1, err_cutoff=0.015
    )

    print("\n\n\nRESULT:\n")
    print_ssa(result)
