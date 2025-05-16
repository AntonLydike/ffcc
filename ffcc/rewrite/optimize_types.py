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

from ffcc.rewrite.rewriter import Rewriter


def is_power_of_two(x: int | float) -> bool:
    return log2(x).is_integer()


def is_close_to_integer(x: int | float, threshold=0.00001) -> bool:
    return abs((round(x) - x) / x) < threshold


def optimize_types(node: IRNode) -> IRNode | None:
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
                m = MathNode(
                    a,
                    c.with_new_value(-amount, IntType(32)),
                    kind=Kind.Ashr,
                    res_type=a.type,
                )
            else:
                m = MathNode(
                    a,
                    c.with_new_value(amount, IntType(32)),
                    kind=Kind.Shl,
                    res_type=a.type,
                )
            if v < 0:
                return MathNode(m, kind=Kind.Negate, res_type=a.type)
            return m
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
            type=res_t,
        ) if is_close_to_integer(v) and isinstance(res_t, IntType):
            return MathNode(
                c.with_new_value(round(v), IntType(width=w)),
                a,
                kind=k,
                res_type=res_t,
            )


types = Rewriter((optimize_types,))
