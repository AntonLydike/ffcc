from ffcc.helper import prod
from ffcc.ir import (
    IRNode,
    MathNode,
    Kind,
    ConstantNode,
    FloatType,
    IntType,
    ConstantLikeNode,
    FoldableNode,
)

from ffcc.opt.rewriter import Rewriter


def factors(n: IRNode) -> list[IRNode]:
    """
    Given a mul, find all factors in it
    """
    if not isinstance(n, MathNode) or not n.kind == Kind.Mul:
        return [n]

    f = []
    for arg in n.argops:
        if isinstance(arg, MathNode) and arg.kind == Kind.Mul:
            f.extend(factors(arg))
        else:
            f.append(arg)
    return f


def simplify_div_exp(node: IRNode) -> IRNode | None:
    """
    Rewrite a / (b^x) to a * b^(-x)
    """
    match node:
        case MathNode(
            kind=Kind.Div,
            argops=(
                lhs,
                MathNode(
                    kind=Kind.Pow,
                    argops=(base, exp),
                ),
            ),
        ):
            return lhs * (base ** (-exp))


def cancel_div(node: IRNode) -> IRNode | None:
    # check for div with
    if (
        not isinstance(node, MathNode)
        or not node.kind == Kind.Div
    ):
        return None

    top = factors(node.argops[0])
    btm = factors(node.argops[1])

    # simple cancellation
    for fac in tuple(btm):
        if fac in top:
            top.remove(fac)
            btm.remove(fac)

    for b in btm:
        if not isinstance(b, MathNode) or not b.kind == Kind.Pow:
            continue
        for t in top:
            if not isinstance(t, MathNode) or not t.kind == Kind.Pow:
                continue
            if t.argops[0] is not b.argops[0]:
                continue
            btm.remove(b)
            top.remove(t)
            top.append(MathNode(
                t.argops[0],
                t.argops[1] - b.argops[1],
                kind=Kind.Pow,
                res_type=t.type
            ))

    cst1 = ConstantNode(1, node.type)

    if top == factors(node.argops[0]) and btm == factors(node.argops[1]):
        return None

    return prod(top, cst1) / prod(btm, cst1)


def div_by_constant(node: IRNode) -> IRNode | None:
    """
    Convert a / c -> c^-1 * a
    """
    match node:
        case MathNode(
            kind=Kind.Div, argops=(a, ConstantLikeNode(value=v) as cst), type=t
        ):
            return cst.with_new_value(1 / v) * a


def neutral_elements(node: IRNode) -> IRNode | None:
    """
    Apply simplifications that arise from neutral elements, e.g.

        1 * x -> x
        x / 1 -> x
        0 * x -> 0

    etc.
    """
    match node:
        # x^0 -> 1
        case MathNode(kind=Kind.Pow, argops=(x, ConstantLikeNode(value=0) as cst)):
            return cst.with_new_value(1)
        # x^1 -> x
        case MathNode(kind=Kind.Pow, argops=(x, ConstantLikeNode(value=1))):
            return x
        # 0x -> 0
        case MathNode(kind=Kind.Mul, argops=(ConstantLikeNode(value=0) as zero, x)):
            return zero
        # 0+x -> x
        case MathNode(kind=Kind.Add, argops=(ConstantLikeNode(value=0), x)):
            return x
        # x - 0 -> x
        case MathNode(kind=Kind.Sub, argops=(x, ConstantLikeNode(value=0))):
            return x
        # 0 - x -> -x
        case MathNode(kind=Kind.Sub, argops=(ConstantLikeNode(value=0), x)):
            return -x
        # 0 / x -> 0
        case MathNode(kind=Kind.Div, argops=(ConstantLikeNode(value=0) as zero, x)):
            return zero
        # x / 1 -> x
        case MathNode(kind=Kind.Div, argops=(x, ConstantLikeNode(value=1))):
            return x
        # 1x -> x
        case MathNode(kind=Kind.Mul, argops=(ConstantLikeNode(value=1), x)):
            return x


def arith(node: IRNode) -> IRNode | None:
    match node:
        # a + (-b) or (-b)+a -> a - b
        case MathNode(
            kind=Kind.Add, argops=(a, MathNode(kind=Kind.Negate, argops=(b,)))
        ) | MathNode(
            kind=Kind.Add, argops=(MathNode(kind=Kind.Negate, argops=(b,)), a)
        ):
            return MathNode(a, b, kind=Kind.Sub, res_type=node.result.type)


def log_identities(node: IRNode) -> IRNode | None:
    match node:
        # specialized on log2(2^x) -> x
        case MathNode(
            kind=Kind.Log2,
            argops=(
                MathNode(
                    kind=Kind.Pow,
                    argops=(ConstantNode(value=2), exp),
                ),
            ),
        ):
            return exp
        # cover log_2(a^x) -> x * log_2(a)
        case MathNode(
            kind=Kind.Log2,
            argops=(
                MathNode(
                    kind=Kind.Pow,
                    args=(base, exp),
                ),
            ),
        ) as log:
            # log2(a^x) -> x * log2(a)
            return MathNode(
                exp,
                MathNode(base, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Mul,
                res_type=log.type,
            )
        # log(a/b) -> log(a) - log(b)
        case MathNode(
            kind=Kind.Log2, argops=(MathNode(kind=Kind.Div, args=(a, b)),)
        ) as log:
            # replace by log(a) - log(b)
            return MathNode(
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                MathNode(b, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Sub,
                res_type=log.type,
            )
        # log(a*b) -> log(a) + log(b)
        case MathNode(
            kind=Kind.Log2, argops=(MathNode(kind=Kind.Mul, args=(a, b)),)
        ) as log:
            # replace by log(a) - log(b)
            return MathNode(
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                MathNode(b, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Add,
                res_type=log.type,
            )
        # exp(a, log_2(x)) -> log_2(a) * x
        case MathNode(
            kind=Kind.Pow, argops=(a, MathNode(kind=Kind.Log2, args=(x,)))
        ) as exp:
            return MathNode(
                x,
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Mul,
                res_type=exp.type,
            )


def symmetry(node: IRNode) -> IRNode | None:
    """
    Apply simplifying rewrites to operations where both sides have the same argument

    convert:
        . x + x -> 2 * x
        . x - x -> 0
        . x / x -> 1
        . -(-x) -> x
    // skip
        . x * x -> x ^ 2
    """
    i32 = IntType(32)
    match node:
        case MathNode(kind, argops=(x, y)) if x is y:
            match kind:
                ## x * x -> x ^ 2
                # case Kind.Mul:
                #    return MathNode(x, ConstantNode(2, i32), kind=Kind.Pow, res_type=x.type)
                # x + x -> x * 2
                case Kind.Add:
                    return ConstantNode(2, i32) * x
                # x - x -> 0
                case Kind.Sub:
                    return ConstantNode(0, x.type)
                # x / x -> 1
                case Kind.Div:
                    return ConstantNode(1, x.type)
        # -(-x) -> x
        case MathNode(
            kind=Kind.Negate, argops=(MathNode(kind=Kind.Negate, argops=(orig,)),)
        ):
            return orig


def constant_shoving(node: IRNode) -> IRNode | None:
    """
    rewrites that move constants around to make them foldable
    """
    match node:
        # switch math(a, const) -> math(const, a)
        case MathNode(kind=k, argops=(a, ConstantLikeNode() as c)) if k in (
            Kind.Mul,
            Kind.Add,
        ) and not isinstance(a, ConstantLikeNode):
            return MathNode(c, a, kind=k, res_type=node.type)
        # a ∘ (c ∘ b) -> c ∘ (a ∘ b)
        # if c is constant and a is not
        case MathNode(
            kind=k1,
            argops=(
                x,
                MathNode(kind=k2, argops=(a, b)),
            )
            | (
                MathNode(kind=k2, argops=(a, b)),
                x,
            ),
            type=t,
        ) if (
            k1 == k2
            and not isinstance(x, ConstantLikeNode)
            and any(isinstance(ab, ConstantLikeNode) for ab in (a, b))
            and k1 in (Kind.Mul, Kind.Add)
        ):
            cst, ncst = (a, b) if isinstance(a, ConstantLikeNode) else (b, a)
            return MathNode(
                cst, MathNode(x, ncst, kind=k1, res_type=t), kind=k1, res_type=t
            )


def constant_fold(node: IRNode) -> IRNode | None:
    match node:
        # generic constant folding of all constant argument foldable op:
        case FoldableNode(
            argops=argops,
            evaluate=evaluate,
            type=res_t,
        ) if all(isinstance(op, ConstantLikeNode) for op in argops):
            vals = [op.value for op in argops]
            result = evaluate(vals)
            if result is not None:
                return ConstantLikeNode.make(result, res_t, argops)
        # c1 ∘ (c2 ∘ x) -> (c1 ∘ c2) -> x
        # for ∘ is + or *
        case MathNode(
            kind=k1,
            argops=(
                ConstantLikeNode(value=v1) as c1,
                MathNode(kind=k2, argops=(ConstantLikeNode(value=v2) as c2, x)),
            ),
            type=res_t,
        ) as math_node if k1 == k2 and k1 in (Kind.Add, Kind.Mul):
            return MathNode(
                ConstantLikeNode.make(math_node.evaluate((v1, v2)), c1.type, (c1, c2)),
                x,
                kind=k1,
                res_type=res_t,
            )
        # c1 * (c2 + (c3 ∘ x))) -> c4 + (c5 ∘ x)
        # ∘ is any operation, c4 = c1 * c2, c5 = c1 * c3
        case MathNode(
            kind=Kind.Mul,
            argops=(
                ConstantLikeNode(value=v1) as c1,
                MathNode(
                    kind=Kind.Add,
                    argops=(
                        ConstantLikeNode(value=v2) as c2,
                        MathNode(
                            kind=k,
                            type=t2,
                            argops=(ConstantLikeNode(value=v3) as c3, x),
                        ),
                    ),
                ),
            ),
        ) if k in (Kind.Add, Kind.Mul, Kind.Div):
            rhs = MathNode(
                ConstantLikeNode.make(v1 * v3, c3.type, (c1, c3)),
                x,
                kind=k,
                res_type=t2,
            )
            return ConstantLikeNode.make(v1 * v2, c2.type, (c1, c2)) + rhs
        # (c1 ∘ (c2 ∘ x)) -> (c1 ∘ c2) ∘ x
        # ∘ is + or *
        case MathNode(
            kind=k1,
            argops=(
                ConstantLikeNode() as c1,
                MathNode(kind=k2, argops=(ConstantLikeNode() as c2, x), type=t2),
            ),
            type=t,
            evaluate=evaluate,
        ) if (
            k1 == k2
            and k1 in (Kind.Mul, Kind.Add)
            and not isinstance(x, ConstantLikeNode)
        ):
            return MathNode(
                ConstantLikeNode.make(evaluate(c1.value, c2.value), t, (c1, c2)),
                x,
                kind=k1,
                res_type=t,
            )


simp = Rewriter(
    (
        simplify_div_exp,
        constant_fold,
        neutral_elements,
        log_identities,
        div_by_constant,
        symmetry,
        arith,
        cancel_div,
        constant_shoving,
    )
)
