import struct
import ctypes
from collections.abc import Sequence
from typing import TypeVar


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


def prod(iter: Sequence[_T], base: _T = 1) -> _T:
    if not iter:
        return base
    base = iter[0]
    for x in iter[1:]:
        base *= x  # pyright: ignore
    return base
