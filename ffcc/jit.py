import os
import ctypes
import tempfile
import subprocess

import numpy as np

from ffcc.print_llvm import print_llvm_func_for, type_to_llvm_type

from ffcc.ir import IRNode, Value, TunableNode

CLANG = os.environ.get('SYNTH_CLANG_BIN', 'clang')

HARNESS_TEMPLATE = """
#include <math.h>

float my_func(float x, {sigma_signature});

int eval_on_domain(float* restrict out, float* restrict domain, int size, {sigma_signature})
{{
    for (int i = 0; i < size; i++) {{
        out[i] = my_func(domain[i], {sigma_args});
    }}
    return 0;
}}

float max_relative_error(float* restrict reference, float* restrict domain, int size, {sigma_signature})
{{
    float max_rel_err = -INFINITY;

    for (int i = 0; i < size; i++) {{
        float res = my_func(domain[i], {sigma_args});
        float rel_err = fabsf((res - reference[i]) / reference[i]);
        max_rel_err = fmax(max_rel_err, rel_err);
    }}

    return max_rel_err;
}}
"""


def instantiate_node_as_jit(node: IRNode, sym_name: str = 'my_func') -> ctypes.CDLL:
    with tempfile.TemporaryDirectory() as tmpdir:

        kernel = os.path.join(tmpdir, 'kernel.ll')
        harness = os.path.join(tmpdir, 'harness.c')
        shared_obj = os.path.join(tmpdir, 'out.so')

        with open(kernel, 'w') as f:
            args = print_llvm_func_for(node, sym_name, f)
        len_tunes = len(args)-1

        with open(harness, 'w') as f:
            print(_build_harness(args), file=f)

        subprocess.check_call(
            [
                CLANG, '-flto=thin', '-lm', kernel, harness, '-O3', '-march=native', '-shared', '-o', shared_obj
            ],
            stderr=subprocess.DEVNULL
        )

        lib = ctypes.CDLL(shared_obj)
        lib.eval_on_domain.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float), # result
            np.ctypeslib.ndpointer(ctypes.c_float), # domain
            ctypes.c_int,
        ] + [ctypes.c_float for _ in range(len_tunes)]
        lib.eval_on_domain.restype = ctypes.c_int

        lib.max_relative_error.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float),
            np.ctypeslib.ndpointer(ctypes.c_float),
            ctypes.c_int,
        ] + [ctypes.c_float for _ in range(len_tunes)]
        lib.max_relative_error.restype = ctypes.c_float

        # set input/output types
        # TODO: handle proper types here
        getattr(lib, sym_name).argtypes = [ctypes.c_float] * len(args)
        getattr(lib, sym_name).restype = ctypes.c_float

        return lib

def _build_harness(args: list[Value]) -> str:
    """
    Assemble the C harness file (from HARNESS_TEMPLATE).

    returns a string that can be written to a file and
    """
    args, tunables = args[:1], args[1:]
    assert all(isinstance(t.owner, TunableNode) for t in tunables)

    return HARNESS_TEMPLATE.format(
        sigma_signature=', '.join('{} sigma_{}'.format(type_to_llvm_type(t.type), i) for i, t in enumerate(tunables)),
        sigma_count=len(tunables),
        sigma_args=', '.join(f'sigma_{i}' for i in range(len(tunables))),
    )
