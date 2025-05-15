from __future__ import annotations

from io import TextIOBase
import ctypes

from ffcc.ir import IRNode, VarNode, TunableNode, Value, Type, ConstantNode, FloatType, IntType, MathNode, Kind, \
    BitCastOperator, CastOperator

_FLOAT_WIDTH_TO_TYPE_NAME = {
    16: "half",
    32: "float",
    64: "double",
    128: "fp128",
}
def _t(typ: Type) -> str:
    """
    Type to llvm type string
    """

    match typ:
        case IntType(width=w):
            return f'i{w}'
        case FloatType(width=w) if w in _FLOAT_WIDTH_TO_TYPE_NAME:
            return _FLOAT_WIDTH_TO_TYPE_NAME[w]
        case _:
            raise ValueError("Cannot convert type to llvm type", typ)


def type_to_llvm_type(t: Type) -> str:
    return _t(t)

def _args(*arg: Value, names: dict[Value, str]) -> str:
    return ", ".join(
        f'{_t(v.type)} {_name(v, names)}' for v in arg
    )

def float_with_bitwidth(value: float, bits: int) -> str:
    if bits == 32:
        return str(ctypes.c_float(value).value)
    elif bits == 64:
        return str(value)


def _name(val: Value, names: dict[Value, str]) -> str:
    # if val is already assigned to a name
    if val in names:
        return names[val]
    # if val.name has no hint, starts with a number,
    elif val.name is None or val.name[0].isdigit():
        name = f'%{len(names)}'
        names[val] = name
        return name

    base_name = f'%{val.name}'
    name = base_name
    used_names = set(names.values())
    i = 1
    while name in used_names:
        name = f"{base_name}_{i}"
        i += 1
    names[val] = name
    return name

def _args_key(arg: Value) -> tuple[int, str]:
    return 0 if isinstance(arg.owner, VarNode) else 1, arg.name

def print_llvm_func_for(node: IRNode, sym_name: str, buff: TextIOBase) -> list[Value]:
    inputs: list[Value] = []

    names: dict[Value, str] = {}

    # grab vars and tunables
    for op in node.walk():
        match op:
            case VarNode(result=r) | TunableNode(result=r):
                inputs.append(r)
                _name(r, names)
            case ConstantNode(result=r, value=v, type=t) if isinstance(t, FloatType):
                names[r] = float_with_bitwidth(v, t.width)
            case ConstantNode(result=r, value=v, type=t) if isinstance(t, IntType):
                names[r] = str(int(v))

    # print function heading
    print(f"define {_t(node.type)} @{sym_name}({_args(*sorted(inputs, key=_args_key), names=names)}) {{", file=buff)

    def ins(result: Value | None = None, *text: str | Value | Type | int | float, indent: str = "  ", res_t: Type | None = None):
        parts = []
        if result is not None:
            parts = [_name(result, names), '=']
        for part in text:
            if isinstance(part, Value):
                parts.append(_name(part, names))
            elif isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Type):
                parts.append(_t(part))
            else:
                parts.append(str(part))
        print(f'{indent}{" ".join(parts)}', file=buff)
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
            raise ValueError('Cannot handle diverging bitwidths yet')
        if isinstance(ty, FloatType):
            res = Value(ty, val.owner, 'cast')
            ins(res, 'sitofp', val_t, val, 'to', ty)
            return res
        elif isinstance(ty, IntType):
            res = Value(ty, val.owner, 'cast')
            ins(res, 'fptosi', val_t, val, 'to', ty)
            return res
        raise ValueError('Unknown type', ty)


    for op in node.walk(reverse=True):
        if op.result in names:
            continue
        match op:
            case MathNode(kind=Kind.Negate, result=r, args=(a,), type=IntType()):
                ins(r, 'sub', a.type, '0,', a, res_t=a.type)
            case MathNode(kind=Kind.Negate, result=r, args=(a,), type=FloatType()):
                ins(r, 'fneg', a.type, a, res_t=a.type)
            case MathNode(kind=k, result=r, args=(a, b)) if k in (Kind.Mul, Kind.Add, Kind.Sub, Kind.Div):
                b = _ensure_type(b, a.type)
                op = k.name.lower()
                if isinstance(a.type, FloatType):
                    op = f'f{op}'
                elif k == Kind.Div:
                    # int div shoudl be sdiv (signed division)
                    op = f's{op}'
                ins(r, op, a.type, a, ',', b, res_t=a.type)
            case MathNode(kind=k, result=r):
                ins(r, 'unknown op', k.name.lower(), res_t=r.type)
            case BitCastOperator(direction=d, args=(a,), result=r, type=ty):
                if d == 'f2i':
                    a = _ensure_type(a, FloatType(a.type.width))
                else:
                    a = _ensure_type(a, IntType(a.type.width))
                ins(r, 'bitcast', a.type, a, 'to', ty)
            case CastOperator(args=(a,), result=r, type=ty):
                a = _ensure_type(a, ty)
                # alias names r and a
                names[r] = _name(a, names)
            case TunableNode() | VarNode() | ConstantNode():
                print("This shouldn't happen!")

    ins(None, 'ret', node.type, node.result)
    print('}', file=buff)

    return inputs