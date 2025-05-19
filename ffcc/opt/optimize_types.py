from ffcc.ir import (
    IRNode,
    MathNode,
    Kind,
    ConstantLikeNode,
    IntType,
    FloatType,
    BitCastOperator,
)
from math import log2

from ffcc.opt.rewriter import Rewriter, RewriteResultModifiedOp


def is_power_of_two(x: int | float) -> bool:
    return log2(x).is_integer()


def is_close_to_integer(x: int | float, threshold=0.00001) -> bool:
    return abs((round(x) - x) / x) < threshold


def optimize_types(node: IRNode) -> IRNode | RewriteResultModifiedOp | None:
    match node:
        # replace a * 2^c with a shift on integer domains
        case MathNode(
            kind=Kind.Mul,
            argops=(
                ConstantLikeNode(value=v) as c,
                a,
            ),
        ) if (
            isinstance(a.type, IntType) and v not in (0, 1) and is_power_of_two(abs(v))
        ):
            amount = int(log2(abs(v)))
            if amount < 0:
                m = a >> c.with_new_value(-amount, IntType(32))
            else:
                m = a << c.with_new_value(amount, IntType(32))
            if v < 0:
                return -m
            return m
        case BitCastOperator(
            direction="f2i",
            argops=(
                MathNode(
                    kind=Kind.Add | Kind.Sub | Kind.Negate, type=IntType(width=w)
                ) as inp,
            ),
        ):
            inp.result.type = FloatType(w)
            return RewriteResultModifiedOp(inp)
        case BitCastOperator(
            direction="i2f",
            argops=(
                MathNode(
                    kind=Kind.Add | Kind.Sub | Kind.Negate, type=FloatType(width=w)
                ) as inp,
            ),
        ):
            inp.result.type = IntType(w)
            return RewriteResultModifiedOp(inp)
        # convert a constant to an int if:
        #  - if it forces a result to be float when it could be int
        #  - that result is used as input to a bitcast
        #  - that constant is pretty close to an int (relative error below 0.00001)
        case MathNode(
            kind=k,
            argops=(
                ConstantLikeNode(value=v, type=FloatType(width=w)) as c,
                a,
            ),
            type=IntType() as int_t,
        ) if is_close_to_integer(v):
            return MathNode(
                c.with_new_value(round(v), IntType(width=w)),
                a,
                kind=k,
                res_type=int_t,
            )


types = Rewriter((optimize_types,))
