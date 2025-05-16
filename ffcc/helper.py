import struct


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
