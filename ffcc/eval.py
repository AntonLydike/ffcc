from collections.abc import Sequence
from typing import Generic, TypeVar
from ffcc.ir import (
    IRNode,
    Value,
    MathNode,
    Kind,
    ConstantNode,
    TunableNode,
    BitCastOperator, ConstantLikeNode,
)
import numpy as np
from ffcc.helper import CASTS

_NP_TYPE_CONV = {
    # i2f
    np.dtype("int32"): np.dtype("float32"),
    np.dtype("int64"): np.dtype("float64"),
    np.dtype("int16"): np.dtype("float16"),
    # f2i
    np.dtype("float32"): np.dtype("int32"),
    np.dtype("float64"): np.dtype("int64"),
    np.dtype("float16"): np.dtype("int16"),
}

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
        case ConstantLikeNode(value=v), ():
            return v
        case MathNode(kind=Kind.Log2), (val,):
            return np.log2(val)
        case MathNode(kind=Kind.Negate), (val,):
            return -val
        case MathNode(kind=Kind.Floor), (val,):
            return np.floor(val)
        case MathNode(kind=Kind.Pow), (base, exp):
            return base**exp
        case MathNode(kind=Kind.Add), (a, b):
            return a + b
        case MathNode(kind=Kind.Sub), (a, b):
            return a - b
        case MathNode(kind=Kind.Mul), (a, b):
            return a * b
        case MathNode(kind=Kind.Div), (a, b):
            return a / b
        case MathNode(kind=Kind.Ashr), (a, b):
            return a // (2**b)
        case MathNode(kind=Kind.Shl), (a, b):
            return a * (2**b)
        case BitCastOperator(direction=d, type=t), (a,):
            val = a
            if isinstance(a, np.ndarray):
                return np.frombuffer(val.tobytes(), _NP_TYPE_CONV[val.dtype])
            else:
                return CASTS[d, t.width](val)
