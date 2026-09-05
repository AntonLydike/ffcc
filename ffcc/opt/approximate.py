from dataclasses import dataclass
from ffcc.ir import (
    IRNode,
    MathNode,
    Kind,
    BitCastOperator,
    TunableNode,
    ConstantNode,
    VarNode,
)
from ffcc.opt.rewriter import RewriteArgs, Rewriter

SIGMA_HINT = 0.0435

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


@dataclass
class Arguments(RewriteArgs):
    exp: bool = False
    log: bool = False
    div: bool = False


def has_var(node: IRNode) -> bool:
    if isinstance(node, VarNode):
        return True
    if any(has_var(child) for child in node.argops):
        return True
    return False


def insert_approximations(node: IRNode, conf: Arguments) -> IRNode | None:
    match node:
        # replace log2(x) -> -B + s2 + s1 / L * I(x)
        case MathNode(
            kind=Kind.Log,
            argops=(xop, ConstantNode(2)),
            result=r,
        ) if has_var(xop) and conf.log:
            x = xop.result
            x_type = x.type
            r_type = r.type
            s1 = TunableNode("s1", 1.0, x_type)
            s2 = TunableNode("s2", SIGMA_HINT, x_type)
            mB = ConstantNode(-B_vals[r_type.width], x_type)
            Linv = ConstantNode(2 ** -L_vals[r_type.width], x_type)
            # return -B + s2 + s1/L * Ix
            return mB + s2 + s1 * Linv * BitCastOperator(x, "f2i")
        # replace b^x -> F((B - s1) * L + x * s2 * L * log_2(b))
        # s1 ~= sigma, s2 ~= 1; both are O(1), so one learning rate works
        # for every float width
        case MathNode(
            kind=Kind.Pow,
            argops=(b, x),
            result=r,
        ) if (has_var(x) or has_var(b)) and conf.exp:
            x_type = x.type
            r_type = r.type
            L = ConstantNode(2 ** L_vals[r_type.width], x_type)
            B = ConstantNode(B_vals[r_type.width], x_type)
            s1 = TunableNode("s1", SIGMA_HINT, x_type)
            s2 = TunableNode("s2", 1.0, x_type)
            logb = MathNode(b, ConstantNode(2, x_type), kind=Kind.Log, res_type=x_type)

            return BitCastOperator(
                direction="i2f",
                value=(B - s1) * L + (x * s2 * L * logb),
            )
        # replace a / x -> a * F(2L * (B - s1) - s2 * I(x))
        case MathNode(kind=Kind.Div, argops=(a, x), result=r) if (
            has_var(x) and conf.div
        ):
            x_type = x.type
            r_type = r.type
            s1 = TunableNode("s1", SIGMA_HINT, x_type)
            s2 = TunableNode("s2", 1.0, x_type)
            twoL = ConstantNode(2 * (2 ** L_vals[r_type.width]), x_type)
            B = ConstantNode(B_vals[r_type.width], x_type)
            return a * BitCastOperator(
                direction="i2f",
                value=(twoL * (B - s1)) - s2 * BitCastOperator(x, "f2i"),
            )


approx = Rewriter[Arguments]((insert_approximations,), Arguments)
