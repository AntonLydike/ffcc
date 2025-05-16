from ffcc.ir import (
    IRNode,
    MathNode,
    FloatType,
    Kind,
    BitCastOperator,
    TunableNode,
    ConstantNode,
    IntType,
    VarNode,
)
from ffcc.rewrite.rewriter import Rewriter

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
        case MathNode(kind=Kind.Log2, args=(x,), result=r) if has_var(x.owner):
            x_type = x.type
            r_type = r.type
            L = 2 ** L_vals[r_type.width]
            B = B_vals[r_type.width]
            sigma = TunableNode("sigma", 0.45066, r_type)
            return MathNode(
                MathNode(
                    MathNode(
                        BitCastOperator(x, "f2i"),
                        ConstantNode(L, x_type),
                        kind=Kind.Div,
                        res_type=x_type,
                    ),
                    ConstantNode(B, x_type),
                    kind=Kind.Add,
                    res_type=x_type,
                ),
                sigma,
                kind=Kind.Add,
                res_type=r_type,
            )
        # replace b^x -> F(L * (B - σ) + x * L / log_b(2))
        # log_b(2) -> log_2(2)/log_2(b) -> 1 / log_2(b)
        # so final formula is: b^x -> F(L * (B - σ) + b * L * log_2(b))
        case MathNode(kind=Kind.Pow, argops=(b, x), result=r) if has_var(x) or has_var(
            b
        ):
            x_type = x.type
            r_type = r.type
            L = 2 ** L_vals[r_type.width]
            B = B_vals[r_type.width]
            sigma = TunableNode("sigma", 0.45066, r_type)
            intt = IntType(max(x_type.width, r_type.width))
            return BitCastOperator(
                direction="i2f",
                value=MathNode(
                    MathNode(  # L * (B - σ)
                        ConstantNode(L, x_type),
                        MathNode(
                            ConstantNode(B, x_type),
                            sigma,
                            kind=Kind.Sub,
                            res_type=x_type,
                        ),
                        kind=Kind.Mul,
                        res_type=x_type,
                    ),
                    MathNode(  # x * L * log_2(b)
                        x,
                        MathNode(
                            ConstantNode(L, x_type),
                            MathNode(b, kind=Kind.Log2, res_type=x_type),
                            kind=Kind.Mul,
                            res_type=x_type,
                        ),
                        kind=Kind.Mul,
                        res_type=x_type,
                    ),
                    kind=Kind.Add,
                    res_type=intt,
                ),
            )
        # replace a / x -> a * F(2L * (B - σ) - I(x))
        case MathNode(kind=Kind.Div, argops=(a, x), result=r) if has_var(x):
            x_type = x.type
            r_type = r.type
            L = 2 ** L_vals[r_type.width]
            B = B_vals[r_type.width]
            sigma = TunableNode("sigma", 0.45066, r_type)
            intt = IntType(max(x_type.width, r_type.width))
            return MathNode(
                a,
                BitCastOperator(
                    direction="i2f",
                    value=MathNode(  # 2L * (B - σ) - I(x)
                        MathNode(
                            ConstantNode(2 * L, x_type),
                            MathNode(
                                ConstantNode(B, x_type),
                                sigma,
                                kind=Kind.Sub,
                                res_type=x_type,
                            ),
                            kind=Kind.Mul,
                            res_type=x_type,
                        ),
                        BitCastOperator(x, "f2i"),
                        kind=Kind.Sub,
                        res_type=intt,
                    ),
                ),
                kind=Kind.Mul,
                res_type=r_type,
            )


approx = Rewriter((insert_approximations,))
