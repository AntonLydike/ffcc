from __future__ import annotations

import math
import sys
from io import TextIOBase
import ctypes

from ffcc.ir import (
    IRNode,
    VarNode,
    TunableNode,
    Value,
    Type,
    ConstantNode,
    FloatType,
    IntType,
    MathNode,
    Kind,
    BitCastOperator,
    CastOperator,
)

_FLOAT_WIDTH_TO_TYPE_NAME = {
    16: "half",
    32: "float",
    64: "double",
    128: "fp128",
}


def _t(typ: Type, vsize: int = 1) -> str:
    """
    Type to llvm type string
    """

    if vsize != 1:
        return f"<{vsize} x {_t(typ, 1)}>"

    match typ:
        case IntType(width=w):
            return f"i{w}"
        case FloatType(width=w) if w in _FLOAT_WIDTH_TO_TYPE_NAME:
            return _FLOAT_WIDTH_TO_TYPE_NAME[w]
        case _:
            raise ValueError("Cannot convert type to llvm type", typ)


def type_to_llvm_type(t: Type, vsize: int = 1) -> str:
    return _t(t, vsize)


def _args(*arg: Value, names: dict[Value, str], vsize: int = 1) -> str:
    return ", ".join(
        f"{_t(v.type, vsize if isinstance(v.owner, VarNode) else 1)} {_name(v, names)}"
        for v in arg
    )


def float_with_bitwidth(value: float, bits: int) -> str:
    if bits == 32:
        return str(ctypes.c_float(value).value)
    elif bits == 64:
        return str(value)
    raise ValueError("Unsupported float width", bits)


def _name(val: Value, names: dict[Value, str]) -> str:
    # if val is already assigned to a name
    if val in names:
        return names[val]
    # if val.name has no hint, starts with a number,
    elif val.name is None or val.name[0].isdigit():
        name = f"%{len(names)}"
        names[val] = name
        return name

    base_name = f"%{val.name}"
    name = base_name
    used_names = set(names.values())
    i = 1
    while name in used_names:
        name = f"{base_name}_{i}"
        i += 1
    names[val] = name
    return name


def _args_key(arg: Value) -> tuple[int, str, float]:
    is_arg = isinstance(arg.owner, VarNode)
    return 0 if is_arg else 1, arg.name, 0 if is_arg else arg.owner.hint


def print_llvm_func_for(
    node: IRNode,
    file: TextIOBase = sys.stdout,
    sym_name: str = "my_func",
    vectorise: int = 1,
    add_scalar: bool = True,
    **kwargs,
) -> tuple[list[Value], list[Value]]:
    seen: set[Value] = set()
    inputs_set: set[Value] = set()

    names: dict[Value, str] = {}

    external_funcs: set[str] = set()

    # grab vars and tunables
    for op in node.walk():
        match op:
            case VarNode(result=r):
                inputs_set.add(r)
            case TunableNode(result=r):
                if r in seen:
                    continue
                seen.add(r)
                if vectorise > 1:
                    fake_val = Value(r.type, op, r.name)
                    fake_val.original = r
                    inputs_set.add(fake_val)
                else:
                    inputs_set.add(r)
            case ConstantNode(result=r, value=v, type=t) if isinstance(t, FloatType):
                if vectorise == 1:
                    names[r] = float_with_bitwidth(v, t.width)
                else:
                    lit = float_with_bitwidth(v, t.width)
                    val_t = _t(t)
                    names[r] = "<{}>".format(", ".join([f"{val_t} {lit}"] * vectorise))
            case ConstantNode(result=r, value=v, type=t) if isinstance(t, IntType):
                if vectorise == 1:
                    names[r] = str(int(v))
                else:
                    val_t = _t(t)
                    names[r] = "<{}>".format(
                        ", ".join([f"{val_t} {str(int(v))}"] * vectorise)
                    )

    inputs = sorted(inputs_set, key=_args_key)

    args_count = sum(isinstance(x.owner, VarNode) for x in inputs)

    # vectorized intrinsics need this prefix to the types
    vec_prefix = ""
    if vectorise > 1:
        vec_prefix = f"v{vectorise}"

    # print function heading
    print(
        f"define {_t(node.type, vectorise)} @{sym_name}({_args(*inputs, names=names, vsize=vectorise)}) {{",
        file=file,
    )

    def ins(
        result: Value | None = None,
        *text: str | Value | Type | int | float,
        indent: str = "  ",
        res_t: Type | None = None,
        to_externals: bool = False,
    ):
        parts = []
        if result is not None:
            parts = [_name(result, names), "="]
        for part in text:
            if isinstance(part, Value):
                parts.append(_name(part, names))
            elif isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Type):
                parts.append(_t(part, vectorise))
            else:
                parts.append(str(part))
        if to_externals:
            external_funcs.add(" ".join(parts))
        else:
            print(f"{indent}{' '.join(parts)}", file=file)
        # cast result to type
        if res_t is not None and result is not None:
            casted_t = _ensure_type(result, result.type, val_t=res_t)
            names[result] = _name(casted_t, names)

    def _ensure_type(val: Value, ty: Type, val_t: Type | None = None) -> Value:
        if val_t is None:
            val_t = val.type

        if val_t == ty:
            return val
        if val_t.width != ty.width:
            raise ValueError(
                "Cannot handle diverging bitwidths yet", str(val_t), str(ty)
            )
        if isinstance(ty, FloatType):
            res = Value(ty, val.owner, "cast")
            ins(res, "sitofp", val_t, val, "to", ty)
            return res
        elif isinstance(ty, IntType):
            res = Value(ty, val.owner, "cast")
            ins(res, "fptosi", val_t, val, "to", ty)
            return res
        raise ValueError("Unknown type", ty)

    # insert the tunable splats:
    if vectorise > 1:
        for arg in inputs:
            if isinstance(arg.owner, VarNode):
                continue
            # temporary value for to use for the vector that contains one value of arg
            vecval = Value(arg.type, arg.owner, arg.name)
            ins(
                vecval,
                f"insertelement",
                arg.type,
                "poison,",
                _t(arg.type),
                arg,
                ",",
                "i64",
                0,
            )
            ins(
                arg.original,
                f"shufflevector",
                arg.type,
                vecval,
                ",",
                arg.type,
                f"poison, <{vectorise} x i32> zeroinitializer",
            )

    for op in node.walk(reverse=True):
        if op.result in names:
            continue
        match op:
            case MathNode(kind=Kind.Negate, result=r, args=(a,), type=IntType()):
                ins(r, "sub", a.type, "0,", a, res_t=a.type)
            case MathNode(kind=Kind.Negate, result=r, args=(a,), type=FloatType()):
                ins(r, "fneg", a.type, a, res_t=a.type)
            case MathNode(kind=k, result=r, args=(a, b)) if k in (
                Kind.Mul,
                Kind.Add,
                Kind.Sub,
                Kind.Div,
                Kind.Shl,
                Kind.Ashr,
            ):
                b = _ensure_type(b, a.type)
                op = k.name.lower()
                if isinstance(a.type, FloatType):
                    op = f"f{op}"
                elif k == Kind.Div:
                    # int div should be sdiv (signed division)
                    op = f"s{op}"
                ins(r, op, a.type, a, ",", b, res_t=a.type)
            case MathNode(
                kind=Kind.Pow,
                result=r,
                args=(
                    base,
                    exp,
                ),
                type=t,
            ):
                base = _ensure_type(base, FloatType(base.type.width))
                if isinstance(exp.type, IntType):
                    # FIXME: powi only supports scalar second arguments, even in vector mode
                    ins(
                        r,
                        "call",
                        f"@llvm.powi.{vec_prefix}{_t(base.type)}.{_t(exp.type)}",
                    )
                    external_funcs.add(
                        f"declare {_t(t, vectorise)} @llvm.powi.{vec_prefix}{str(base.type)}.{str(exp.type)}({_t(base.type, vectorise)}, {_t(exp.type)})"
                    )
                else:
                    exp = _ensure_type(exp, base.type)
                    ins(
                        r,
                        "call",
                        t,
                        f"@llvm.pow.{str(base.type)}(",
                        base.type,
                        base,
                        ",",
                        exp.type,
                        exp,
                        ")",
                    )
                    external_funcs.add(
                        f"declare {_t(t, vectorise)} @llvm.pow.{vec_prefix}{str(base.type)}({_t(base.type, vectorise)}, {_t(exp.type, vectorise)})"
                    )
            case MathNode(
                kind=Kind.Log,
                result=r,
                argops=(a, ConstantNode(base)),
                type=FloatType() as ft,
            ) if base in (math.e, 2, 10):
                basestr = {math.e: "", 10: "10", 2: "2"}
                a = _ensure_type(a.result, ft)
                intr = f"@llvm.log{basestr[base]}.{vec_prefix}{str(ft)}"
                ins(r, "call", ft, f"{intr}(", a.type, a, ")")
                external_funcs.add(
                    f"declare {_t(ft, vectorise)} {intr}({_t(a.type, vectorise)})"
                )
            case MathNode(
                kind=Kind.Log,
                result=r,
                argops=(a, ConstantNode(base)),
                type=FloatType() as ft,
            ):
                # FIXME: implement arbitrary bases
                raise NotImplementedError(
                    "Only log base e, 2 and 10 are supported in llvm backend"
                )
            case MathNode(kind=k, result=r):
                ins(r, "unknown op", k.name.lower(), res_t=r.type)
            case BitCastOperator(direction=d, args=(a,), result=r, type=ty):
                if d == "f2i":
                    a = _ensure_type(a, FloatType(a.type.width))
                else:
                    a = _ensure_type(a, IntType(a.type.width))
                ins(r, "bitcast", a.type, a, "to", ty)
            case CastOperator(args=(a,), result=r, type=ty):
                a = _ensure_type(a, ty)
                # alias names r and a
                names[r] = _name(a, names)
            case TunableNode() | VarNode() | ConstantNode():
                print("This shouldn't happen!")

    ins(None, "ret", node.type, node.result)
    print("}\n", file=file)
    print("\n".join(external_funcs), file=file)

    if add_scalar and vectorise > 1:
        print_llvm_func_for(
            node,
            file,
            sym_name=f"{sym_name}_scalar",
            vectorise=1,
            add_scalar=False,
            **kwargs,
        )

    return inputs[:args_count], inputs[args_count:]
