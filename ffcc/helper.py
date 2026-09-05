import struct
import ctypes
from collections.abc import Sequence
from typing import TypeVar

import numpy as np


# casting operators
def i2f(i: int) -> float:
    return struct.unpack("f", struct.pack("i", ctypes.c_int32(i).value))[0]


def f2i(f: float) -> int:
    return struct.unpack("i", struct.pack("f", float(f)))[0]


# casting operators
def i2f_64(i: int) -> float:
    return struct.unpack("d", struct.pack("q", ctypes.c_int32(i).value))[0]


def f2i_64(f: float) -> int:
    return struct.unpack("q", struct.pack("d", float(f)))[0]


CASTS = {
    ("f2i", 32): f2i,
    ("i2f", 32): i2f,
    ("f2i", 64): f2i_64,
    ("i2f", 64): i2f_64,
}

_T = TypeVar("_T")


# numpy dtypes for the IEEE-754 widths we support
_FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64}


def format_float_width(value: float, width: int) -> str:
    """
    Format a float as the shortest decimal string that round-trips to the
    same value at the given IEEE-754 bit width, i.e. the value a constant of
    that width actually holds at runtime. Unknown widths fall back to repr.
    """
    dt = _FLOAT_DTYPES.get(width)
    if dt is None:
        return repr(value)
    with np.errstate(over="ignore", invalid="ignore"):
        return str(dt(value))


def prod(iter: Sequence[_T], base: _T = 1) -> _T:
    if not iter:
        return base
    base = iter[0]
    for x in iter[1:]:
        base *= x  # pyright: ignore
    return base
