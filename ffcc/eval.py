from collections.abc import Sequence
from typing import TypeVar
from ffcc.ir import (
    IRNode,
    Value,
    MathNode,
    Kind,
    BitCastOperator,
    ConstantLikeNode,
    CastOperator,
    IntType,
)
import numpy as np
from ffcc.helper import CASTS

T = TypeVar("T", bound=np.ndarray | float)


def evaluate(node: IRNode, assignment: dict[Value, T]) -> T:
    try:
        for p in node.walk(reverse=True):
            if p.result in assignment:
                continue
            assignment[p.result] = evaluate_node(p, [assignment[arg] for arg in p.args])

        return assignment[node.result]
    except TypeError:
        print("Error at", assignment)
        raise


def partial_evaluation(node: IRNode, assignment: dict[Value, T]) -> T:
    # evaluate args:
    args = [
        (
            assignment[arg]
            if arg in assignment
            else partial_evaluation(arg.owner, assignment)
        )
        for arg in node.args
    ]
    return evaluate_node(node, args)


def evaluate_node(node: IRNode, vals: Sequence[T | float]) -> T | float:
    """
    Evaluate the result of node, assuming assigned values
    """
    match node, vals:
        case ConstantLikeNode(value=val), ():
            return val
        case MathNode(kind=Kind.Log2), (val,):
            return np.log2(val)
        case MathNode(kind=Kind.Negate), (val,):
            return -val
        case MathNode(kind=Kind.Floor), (val,):
            return np.floor(val)
        case MathNode(kind=Kind.Pow), (base, exp):
            return np.pow(base, exp)
        case MathNode(kind=Kind.Add), (a, b):
            return np.add(a, b)
        case MathNode(kind=Kind.Sub), (a, b):
            return np.subtract(a, b)
        case MathNode(kind=Kind.Mul), (a, b):
            return np.multiply(a, b)
        case MathNode(kind=Kind.Div), (a, b):
            return np.divide(a, b)
        case MathNode(kind=Kind.Ashr), (a, b):
            return np.floor_divide(a, np.pow(2, b))
        case MathNode(kind=Kind.Shl), (a, b):
            return np.multiply(a, np.pow(2, b))
        case BitCastOperator(direction=d, type=t), (a,) if isinstance(a, (int, float)):
            return CASTS[d, t.width](a)
        case BitCastOperator(direction="f2i", type=t), (a,):
            buff = a.astype(np.dtype(f"float{t.width}")).tobytes()
            return np.frombuffer(buff, np.dtype(f"int{t.width}"))
        case BitCastOperator(direction="i2f", type=t), (a,):
            buff = a.astype(np.dtype(f"int{t.width}")).tobytes()
            return np.frombuffer(buff, np.dtype(f"float{t.width}"))
        case CastOperator(type=t), (val,) if isinstance(val, np.ndarray):
            if isinstance(t, IntType):
                dt = np.dtype(f"int{t.width}")
            else:
                dt = np.dtype(f"float{t.width}")
            return val.astype(dt)
        case CastOperator(type=t), (val,):
            if isinstance(t, IntType):
                return int(val)
            return float(val)
