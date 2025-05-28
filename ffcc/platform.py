import subprocess


def has_comp_flag(flag: str) -> bool:
    res = subprocess.check_output(
        ["clang", "-E", "-", "-march=native", "-###"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return f"+{flag}" in res
