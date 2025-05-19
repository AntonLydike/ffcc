from ffcc.ir import IRNode, VarNode, ConstantNode, FloatType, MathNode, Kind, Value
import math

f32 = FloatType(32)


def cst(x: float | int) -> ConstantNode:
    return ConstantNode(x, f32)


def log2(n: IRNode | Value) -> MathNode:
    return MathNode(n, kind=Kind.Log2, res_type=n.type)


def ln(n: IRNode | Value) -> IRNode:
    return cst(1 / math.log2(math.e)) * log2(n)


def diff(node: IRNode, var: VarNode) -> IRNode:
    if node == var:
        return ConstantNode(value=1, type=f32)
    if var not in node:
        return ConstantNode(value=0, type=f32)
    match node:
        # floor(x)' -> 0
        case MathNode(kind=Kind.Floor):
            return ConstantNode(0, f32)
        # (-a)' -> -a'
        case MathNode(kind=Kind.Negate, argops=(a,), type=t):
            return -diff(a, var)
        # (x^a)' -> ax^(a-1)
        case MathNode(kind=Kind.Pow, argops=(x, a), type=t) if (
            x == var and var not in a
        ):
            a = a.copy()
            x = x.copy()
            return a * (x ** (a - ConstantNode(1, t)))
        # a+b -> a'+b' (same for sub)
        case MathNode(kind=k, argops=(a, b), type=t) if k in (Kind.Add, Kind.Sub):
            return diff(a, var) + diff(b, var)
        # product rule:
        # (ab)' -> a'b + ab'
        case MathNode(kind=Kind.Mul, argops=(a, b), type=t):
            return diff(b, var) * b.copy() + diff(a, var) * a.copy()
        # quotient rule:
        # (a/b)' -> (a'b - ab')/(b^2)
        case MathNode(kind=Kind.Div, argops=(a, b), type=t):
            ad = diff(a, var)
            bd = diff(b, var)
            b = b.copy()
            a = a.copy()
            return (ad * b - a * bd) / (b ** ConstantNode(2, t))
        # (b^e)' -> ln(b)b^e * e'
        case MathNode(kind=Kind.Pow, argops=(base, exp)) if var not in base:
            cbase = base.copy()
            cexp = exp.copy()
            return ln(cbase) * (cbase**cexp) * diff(exp, var)

        # chain rule:
        # h(g(x))' -> h'(g(x)) * g'(x)
    raise NotImplementedError()
