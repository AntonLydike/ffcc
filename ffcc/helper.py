import struct
import ctypes
from collections.abc import Sequence
from typing import Iterable

import math
import time
from math import floor, log, pow


# casting operators
def i2f(i: int) -> float:
    return struct.unpack("f", struct.pack("i", ctypes.c_int32(i).value))[0]


def f2i(f: float) -> int:
    return struct.unpack("i", struct.pack("f", f))[0]


# casting operators
def i2f_64(i: int) -> float:
    return struct.unpack("d", struct.pack("q", ctypes.c_int32(i).value))[0]


def f2i_64(f: float) -> int:
    return struct.unpack("q", struct.pack("d", f))[0]


CASTS = {
    ("f2i", 32): f2i,
    ("i2f", 32): i2f,
    ("i2f_64", 32): i2f_64,
    ("f2i_64", 32): f2i_64,
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


def duration(dur_in_seconds: float) -> str:
    """
    Convert a duration in seconds to a string

    :param dur_in_seconds:
    :return:
    """
    if dur_in_seconds <= 0:
        return "0s"
    if dur_in_seconds < 60:
        exp = floor(log(dur_in_seconds, 1000))
        if exp not in _dur_to_prefix_small:
            return f"{dur_in_seconds}s"
        ttl = dur_in_seconds / pow(1000, exp)
        sfx = _dur_to_prefix_small[exp]
        return f"{ttl:.3g}{sfx}".lstrip()

    last_num, last_suf = 0, "ms"
    num, suf = dur_in_seconds, "s"
    for div, new_suf in _time_divisions:
        if num < div:
            if last_num < 1:
                if round(num) == num:
                    return f"{round(num)}{suf}"
                return f"{num:.3g}{suf}"
            return f"{floor(num)}{suf}{round(last_num)}{last_suf}"
        last_suf = suf
        last_num = num % div
        num /= div
        suf = new_suf

    if last_num < 1:
        if round(num) == num:
            num = round(num)
        return f"{num:.3g}{suf}"
    return f"{round(num)}{suf}{round(last_num)}{last_suf}"


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


def print_progress(current: int, total: int, start_time: float, message: str = ""):
    percent = 100 * current // total
    elapsed = time.time() - start_time
    avg_time = elapsed / (current + 1)
    remaining = duration(avg_time * (total - current - 1))

    bar_len = 50
    filled_len = int(bar_len * percent // 100)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    size = math.ceil(math.log10(total))

    print(
        f"\r[{bar}] {percent}% ({current:0{size}}/{total}) - ETA: {remaining} ({duration(avg_time)}/elem) {message}   ",
        flush=True,
        end="",
    )


def prod(iter: Sequence, base=1):
    if not iter:
        return base
    base = iter[0]
    for x in iter[1:]:
        base *= x
    return base
