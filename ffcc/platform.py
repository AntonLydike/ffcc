import subprocess


def has_comp_flag(flag: str) -> bool:
    res = subprocess.check_output(
        ["clang", "-E", "-", "-march=native", "-###"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return f"+{flag}" in res


def vector_width_bits() -> int:
    if has_comp_flag("avx512"):
        return 512
    elif has_comp_flag("avx2"):
        return 256
    elif has_comp_flag("sse"):
        return 128
    return 64
