from ffcc.ir import (
    IRNode,
    MathNode,
    Kind,
    BitCastOperator,
    TunableNode,
    ConstantNode,
    VarNode,
)
from ffcc.opt.rewriter import Rewriter

SIGMA_HINT = 0.0450466

L_vals = {
    16: 10,
    32: 23,
    64: 52,
    128: 112,
    256: 236,
}
"""
Map float width to mantissa width for IEEE floating point formats (binary16-256)
"""

B_vals = {
    16: 15,
    32: 127,
    64: 1023,
    128: 16383,
    256: 262143,
}
"""
Bias for different IEEE floating point formats (binary16-256)
"""


def has_var(node: IRNode) -> bool:
    if isinstance(node, VarNode):
        return True
    if any(has_var(child) for child in node.argops):
        return True
    return False


def insert_approximations(node: IRNode) -> IRNode | None:
    match node:
        # replace log2(x) -> I(x)/L - B + σ
        case MathNode(
            kind=Kind.Log, argops=(xop, ConstantNode(2)), result=r
        ) if has_var(xop):
            x = xop.result
            x_type = x.type
            r_type = r.type
            Linv = ConstantNode(2 ** -L_vals[r_type.width], x_type)
            mB = ConstantNode(-B_vals[r_type.width], x_type)
            sigma = TunableNode("sigma", SIGMA_HINT, r_type)
            # mb = -B
            # Linv = 1/L
            # return -B + σ + 1/L * Ix
            return mB + sigma + Linv * BitCastOperator(x, "f2i")
        # replace b^x -> F(L * (B - σ) + x * L / log_b(2))
        # log_b(2) -> log_2(2)/log_2(b) -> 1 / log_2(b)
        # so final formula is: b^x -> F(L * (B - σ) + x * L * log_2(b))
        case MathNode(
            kind=Kind.Pow,
            argops=(b, x),
            result=r,
        ) if has_var(
            x
        ) or has_var(b):
            x_type = x.type
            r_type = r.type
            L = ConstantNode(2 ** L_vals[r_type.width], x_type)
            B = ConstantNode(B_vals[r_type.width], x_type)
            sigma = TunableNode("sigma", SIGMA_HINT, r_type)
            return BitCastOperator(
                direction="i2f",
                value=(L * (B - sigma))
                + (
                    L
                    * MathNode(
                        b, ConstantNode(2, x_type), kind=Kind.Log, res_type=x_type
                    )
                    * x
                ),
            )
        # replace a / x -> a * F(2L * (B - σ) - I(x))
        case MathNode(kind=Kind.Div, argops=(a, x), result=r) if has_var(x):
            x_type = x.type
            r_type = r.type
            twoL = ConstantNode(2 * (2 ** (L_vals[r_type.width])), x_type)
            B = ConstantNode(B_vals[r_type.width], x_type)
            sigma = TunableNode("sigma", SIGMA_HINT, r_type)
            return a * BitCastOperator(
                direction="i2f", value=twoL * (B - sigma) - BitCastOperator(x, "f2i")
            )


approx = Rewriter((insert_approximations,))
