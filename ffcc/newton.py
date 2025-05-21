from ffcc.diff import diff
from ffcc.ir import IRNode, VarNode


def var_of(node: IRNode) -> VarNode:
    var = None
    for n in node.walk():
        if isinstance(n, VarNode):
            var = n
            break
    if var is None:
        raise ValueError("No variable found")
    return var


def newton(f: IRNode) -> IRNode:
    var = var_of(f)

    # find a function that has a zero at f(x)
    # this can be done by finding a function g(y) = x (solving y = f(x) for x)
    # This leaves g(y) - x = 0

    df = diff(f, var)
    df_of_x = df.subs({var_of(df): f})

    f = f.copy()
    f_of_x = f.subs({var_of(f): f})

    return f - f_of_x / df_of_x
