import struct
import ctypes
import sys
from collections.abc import Sequence
from aalib.multilines import MultilineCtx

import math
import time
from aalib.duration import duration


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

_dur_to_prefix_small = {
    0: "s",
    -1: "ms",
    -2: "μs",
    -3: "ns",
    -4: "ps",
    -5: "fs",
    -6: "as",
    -7: "zs",
}
_time_divisions = (
    (60, "m"),
    (60, "h"),
    (24, "d"),
    (365.25 / 12, "mo"),
    (12, "y"),
    (100, "century"),
)



def step_float(x, step: int):
    x = float(x)

    if math.isnan(x) or math.isinf(x):
        return x

    if x == 0.0:
        x = 0.0

    sign = 1
    if x < 0.0:
        sign = -1
        x = -x

    n: int = struct.unpack("<I", struct.pack("<f", x))[0]
    n += step

    # apply sign switches:
    if n < 0:
        n = -n
        sign *= -1

    if n >= (1 << 31):
        return sign * math.inf

    return sign * struct.unpack("<f", struct.pack("<I", n))[0]



def prod(iter: Sequence, base=1):
    if not iter:
        return base
    base = iter[0]
    for x in iter[1:]:
        base *= x
    return base



def speedometer(t0: float, iters: int, message: str = "", file = sys.stdout):
    elem_per_s = iters / (time.time() - t0)
    if elem_per_s < 1:
        s_per_elem = 1 / elem_per_s
        print(f"{duration(s_per_elem)}/elem ({iters} elems processed) {message}", end="\r", file=file)
    if elem_per_s > 1000:
        elem_per_s = int(elem_per_s)
    print(f"{elem_per_s}elem/s ({iters} elems processed) {message}", end="\r", file=file)
