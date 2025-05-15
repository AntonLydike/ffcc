from ffcc.ir import IRNode, MathNode, Kind, ConstantNode, FloatType, IntType, CastOperator, BitCastOperator
import math
import struct
import ctypes

from ffcc.rewrite.rewriter import Rewriter

# casting operators
def i2f(i: int) -> float:
    return struct.unpack('f', struct.pack('i', ctypes.c_int32(i).value))[0]


def f2i(f: float) -> int:
    return struct.unpack('i', struct.pack('f', f))[0]


# casting operators
def i2f_64(i: int) -> float:
    return struct.unpack('d', struct.pack('q', ctypes.c_int32(i).value))[0]


def f2i_64(f: float) -> int:
    return struct.unpack('q', struct.pack('d', f))[0]


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
                    args=(base, exp),
                ) as exp_node
            )
        ) as div:
            return MathNode(
                lhs,
                MathNode(
                    base,
                    MathNode(exp, kind=Kind.Negate, res_type=exp.type),
                    kind=Kind.Pow,
                    res_type=exp_node.result.type
                ),
                kind=Kind.Mul,
                res_type=div.result.type
            )


def div_by_constant(node: IRNode) -> IRNode | None:
    match node:
        case MathNode(
            kind=Kind.Div,
            argops=(a, ConstantNode(value=v, type=ct)),
            type=t
        ):
            return MathNode(
                ConstantNode(value=1/v, type=ct),
                a,
                kind=Kind.Mul,
                res_type=t
            )


def constant_fold(node: IRNode) -> IRNode | None:
    match node:
        case MathNode(
            kind,
            argops,
            results=(r,)
        ) if all(isinstance(op, ConstantNode) for op in argops):
            vals = [op.value for op in argops]
            res_t = r.type
            match kind:
                case Kind.Pow:
                    return ConstantNode(vals[0] ** vals[1], res_t)
                case Kind.Mul:
                    return ConstantNode(vals[0] * vals[1], res_t)
                case Kind.Div:
                    return ConstantNode(vals[0] / vals[1], res_t)
                case Kind.Add:
                    return ConstantNode(vals[0] + vals[1], res_t)
                case Kind.Sub:
                    return ConstantNode(vals[0] - vals[1], res_t)
                case Kind.Floor:
                    return ConstantNode(math.floor(vals[0]), IntType(argops[0].result.type.width))
                case Kind.Negate:
                    return ConstantNode(-vals[0], res_t)
                case Kind.Log2:
                    return ConstantNode(math.log2(vals[0]), res_t)
        case CastOperator(argops=(ConstantNode(value=v)), type=t):
            if isinstance(t, IntType):
                # FIXME: properly truncate bits
                return ConstantNode(int(v), t)
            elif isinstance(t, FloatType):
                return ConstantNode(float(v), t)
        case BitCastOperator(argops=(ConstantNode(value=val)), direction=d, type=t) if t.width == 32:
            if d == 'f2i':
                return ConstantNode(f2i(float(val)), t)
            elif d == 'i2f':
                return ConstantNode(i2f(int(val)), t)
        case BitCastOperator(argops=(ConstantNode(value=val)), direction=d, type=t) if t.width == 64:
            if d == 'f2i':
                return ConstantNode(f2i_64(float(val)), t)
            elif d == 'i2f':
                return ConstantNode(i2f_64(int(val)), t)


def neutral_elements(node: IRNode) -> IRNode | None:
    match node:
        case MathNode(
            kind,
            argops,
        ) if any(isinstance(op, ConstantNode) for op in argops) and len(argops) == 2:
            v0, v1 = argops
            match (v0, kind, v1):
                case (ConstantNode(value=0), kind.Add, b) | (b, kind.Add, ConstantNode(value=0)) | (b, kind.Sub, ConstantNode(value=0)):
                    return b
                case (ConstantNode(value=1), kind.Mul, b) | (b, Kind.Mul, ConstantNode(value=1)) | (b, kind.Div, ConstantNode(value=1)):
                    return b
                case (ConstantNode(value=0), kind.Sub, b):
                    return MathNode(b, kind=Kind.Negate, res_type=b.type)
        # add a -b or add -b a -> sub a b
        case MathNode(kind=Kind.Add, argops=(a, MathNode(kind=Kind.Negate, argops=(b,)))) | MathNode(kind=Kind.Add, argops=(MathNode(kind=Kind.Negate, argops=(b,)), a)):
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
            )
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
            )
        ) as log:
            # log2(a^x) -> x * log2(a)
            return MathNode(
                exp,
                MathNode(base, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Mul,
                res_type=log.type
            )
        # log(a/b) -> log(a) - log(b)
        case MathNode(
            kind=Kind.Log2,
            argops=(
                MathNode(kind=Kind.Div, args=(a, b)),
            )
        ) as log:
            # replace by log(a) - log(b)
            return MathNode(
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                MathNode(b, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Sub,
                res_type=log.type
            )
        # log(a*b) -> log(a) + log(b)
        case MathNode(
            kind=Kind.Log2,
            argops=(
                MathNode(kind=Kind.Mul, args=(a, b)),
            )
        ) as log:
            # replace by log(a) - log(b)
            return MathNode(
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                MathNode(b, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Add,
                res_type=log.type
            )
        # exp(a, log_2(x)) -> log_2(a) * x
        case MathNode(
            kind=Kind.Pow,
            argops=(a, MathNode(kind=Kind.Log2, args=(x,)))
        ) as exp:
            return MathNode(
                x,
                MathNode(a, kind=Kind.Log2, res_type=FloatType(32)),
                kind=Kind.Mul,
                res_type=exp.type
            )

def symmetry(node: IRNode) -> IRNode | None:
    """
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
                #case Kind.Mul:
                #    return MathNode(x, ConstantNode(2, i32), kind=Kind.Pow, res_type=x.type)
                # x + x -> x * 2
                case Kind.Add:
                    return MathNode(ConstantNode(2, i32), x, kind=Kind.Mul, res_type=x.type)
                # x - x -> 0
                case Kind.Sub:
                    return ConstantNode(0, x.type)
                # x / x -> 1
                case Kind.Div:
                    return ConstantNode(1, x.type)
        # -(-x) -> x
        case MathNode(kind=Kind.Negate, argops=(MathNode(kind=Kind.Negate, argops=(orig,)),)):
            return orig

def constant_shoving(node: IRNode) -> IRNode | None:
    """
    shove constants to the left
    """
    match node:
        case MathNode(
            kind=k,
            argops=(a, ConstantNode() as c)
        ) if k in (Kind.Mul, Kind.Add) and not isinstance(a, ConstantNode):
            return MathNode(c, a, kind=k, res_type=node.type)
        # (c1 + (c2 + x)) -> (c1 + c2) + x
        case MathNode(
            kind=k1,
            argops=(ConstantNode() as c1, MathNode(
                kind=k2,
                argops=(ConstantNode() as c2, x),
                type=t2
            )),
            type=t
        ) if k1 == k2 and k1 in (Kind.Mul, Kind.Add) and t2 == t and not isinstance(x, ConstantNode):
            return MathNode(
                MathNode(
                    c1, c2, kind=k1, res_type=t
                ),
                x,
                kind=k1,
                res_type=t
            )

simp = Rewriter((
    simplify_div_exp,
    constant_fold,
    neutral_elements,
    log_identities,
    div_by_constant,
    symmetry,
    constant_shoving,
))
