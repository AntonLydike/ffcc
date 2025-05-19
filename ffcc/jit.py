import logging
import sys
from io import StringIO

from typing import Sequence

import os
import shlex
import ctypes
import tempfile
import subprocess

import numpy as np

from ffcc.print_llvm import print_llvm_func_for, type_to_llvm_type

from ffcc.ir import IRNode, Value, TunableNode, VarNode, FloatType

LOGGER = logging.getLogger(__name__)

CLANG = os.environ.get("SYNTH_CLANG_BIN", "clang")

DEBUG_JIT_ENABLE = os.environ.get("FFCC_DEBUG_JIT", None)
"""
Allow debugging the jit
"""

HARNESS_TEMPLATE = """#include <math.h>
#include <stdint.h>
{headers}

typedef int32_t i32;
typedef int64_t i64;

float my_func(float x{sigma_signature});

int eval_on_domain(float* restrict out, float* restrict domain, int64_t size{sigma_signature})
{{

    {omp_pragma}
    for (int64_t i = 0; i < size; i++) {{
        out[i] = my_func(domain[i]{sigma_args});
    }}
    return 0;
}}

float max_relative_error(float* restrict reference, float* restrict domain, int64_t size{sigma_signature})
{{
    float max_rel_err = -INFINITY;

    {omp_pragma}
    for (int64_t i = 0; i < size; i++) {{
        float res = my_func(domain[i]{sigma_args});
        float rel_err = fabsf((res - reference[i]) / reference[i]);
        max_rel_err = fmax(max_rel_err, rel_err);
    }}

    return max_rel_err;
}}

int64_t sweep_tunables(float* restrict output, float* restrict reference, float* restrict domain, int64_t size{sigma_signature}{sigma_signature_max}{sigma_signature_steps})
{{
    int64_t res_idx = 0;
    
    {omp_pragma_sweep}
{sigma_loops}

        output[res_idx++] = max_relative_error(reference, domain, size{sigma_sweep_args});
    
    {sigma_loops_end}

    return res_idx;
}}
"""


class Program:
    tunables: list[TunableNode]
    variables: list[VarNode]
    dll: ctypes.CDLL

    def __init__(self, node: IRNode):
        dll, args, tunes = instantiate_node_as_jit(node)

        self.variables = [a.owner for a in args]
        self.dll = dll
        self.tunables = [t.owner for t in tunes]

        assert len(self.variables) == 1, "Cannot handle multi-variable programs yet!"

    @property
    def initial_tune(self) -> tuple[float, ...]:
        return tuple(t.hint for t in self.tunables)

    def eval_on_domain(
        self,
        domain: np.ndarray,
        tunables: Sequence[float | int],
        result: np.ndarray | None = None,
    ) -> np.ndarray:
        if result is None:
            result = np.zeros_like(domain)
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        if self.dll.eval_on_domain(result, domain, domain.size, *tunables) != 0:
            raise RuntimeError("evaulation failed")
        return result

    def max_relative_error(
        self, reference: np.ndarray, domain: np.ndarray, tunables: Sequence[float]
    ) -> float:
        assert reference.size == domain.size
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        return self.dll.max_relative_error(reference, domain, domain.size, *tunables)

    def sweep_tunables(
        self,
        reference: np.ndarray,
        domain: np.ndarray,
        tunable_range: Sequence[tuple[float, float]],
        tunable_steps: Sequence[int],
        result: np.ndarray | None = None,
    ) -> np.ndarray:
        assert (
            len(tunable_range) == len(tunable_steps) == len(self.tunables)
        ), "tunable ranges and steps must match number of tunables"
        if result is None:
            result = np.zeros(tunable_steps, dtype=reference.dtype)
        assert result.size == np.prod(tunable_steps), "Result must fit all outputs"
        return result

    def __call__(self, x: float, *tunables: float | int) -> float:
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        return self.dll.my_func(x, *tunables)


def instantiate_node_as_jit(
    node: IRNode, omp_threads: int = 4, lto: bool = True, opt_level: int = 3
) -> tuple[ctypes.CDLL, list[Value], list[Value]]:
    if DEBUG_JIT_ENABLE is None:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
    elif DEBUG_JIT_ENABLE == "":
        tmpdir_obj = tempfile.TemporaryDirectory(suffix="persistent", delete=False)
        tmpdir = tmpdir_obj.name
        LOGGER.info(f"Created persistent JIT dir: {tmpdir}")
    else:
        tmpdir = DEBUG_JIT_ENABLE

    kernel = os.path.join(tmpdir, "kernel.ll")
    harness = os.path.join(tmpdir, "harness.c")
    shared_obj = os.path.join(tmpdir, "out.so")

    out_t = node.type.ctype

    # do not re-generate files if jit debugging is enabled and files exist
    if not os.path.exists(kernel):
        with open(kernel, "w") as f:
            args, tunes = print_llvm_func_for(node, "my_func", f)
    else:
        args, tunes = print_llvm_func_for(node, "my_func", StringIO())

    # FIXME: allow programs with signatures other than "x: f32"
    assert len(args) == 1
    input_t = args[0].type.ctype

    # do not re-generate files if jit debugging is enabled and files exist
    if not os.path.exists(harness):
        with open(harness, "w") as f:
            print(_build_harness(args, tunes, omp_threads=omp_threads), file=f)

    cflags = ["-march=native", "-fuse-ld=lld", "-Wno-override-module"]
    if omp_threads > 1:
        cflags.append("-fopenmp")
    if lto:
        cflags.append("-flto=thin")
    if opt_level > 0:
        cflags.append(f"-O{opt_level}")

    # make sure to read nixos added cflags
    if "NIX_CFLAGS_COMPILE_FOR_TARGET" in os.environ:
        cflags += shlex.split(os.environ["NIX_CFLAGS_COMPILE"])

    # make sure to read nixos added ldflags
    if "NIX_LDFLAGS_FOR_TARGET" in os.environ:
        cflags += shlex.split(os.environ["NIX_LDFLAGS_FOR_TARGET"])

    command = (CLANG, *cflags, "-lm", kernel, harness, "-shared", "-o", shared_obj)
    LOGGER.info(f'Compile command: {" ".join(command)}')

    if DEBUG_JIT_ENABLE is not None:
        input("JIT input files generated, press ENTER to continue compilation...")

    subprocess.check_call(
        command,
        stderr=sys.stderr if LOGGER.level <= logging.INFO else subprocess.DEVNULL,
    )

    lib = ctypes.CDLL(shared_obj)

    # set types for eval_on_domain
    lib.eval_on_domain.argtypes = [
        np.ctypeslib.ndpointer(out_t),  # result
        np.ctypeslib.ndpointer(input_t),  # domain
        ctypes.c_int64,  # domain.size
    ] + [
        t.type.ctype for t in tunes  # tunables
    ]  # *sigma

    lib.eval_on_domain.restype = ctypes.c_int

    # set types for max_relative_error
    lib.max_relative_error.argtypes = [
        np.ctypeslib.ndpointer(input_t),  # reference
        np.ctypeslib.ndpointer(input_t),  # domain
        ctypes.c_int64,  # domain.size = referece.size
    ] + [
        t.type.ctype for t in tunes  # tunables
    ]  # *sigma

    lib.max_relative_error.restype = ctypes.c_float

    # set types for sweep_tunables
    lib.sweep_tunables.argtypes = (
        [
            np.ctypeslib.ndpointer(ctypes.c_float),  # output of sweep
            np.ctypeslib.ndpointer(input_t),  # reference
            np.ctypeslib.ndpointer(input_t),  # domain
            ctypes.c_int64,  # domain.size = referece.size
        ]
        + [t.type.ctype for t in tunes]
        + [t.type.ctype for t in tunes]
        + [ctypes.c_int for _ in tunes]
    )

    lib.sweep_tunables.restype = ctypes.c_int64

    # set input/output types
    # TODO: handle proper types here
    lib.my_func.argtypes = [input_t, *(t.type.ctype for t in tunes)]
    lib.my_func.restype = out_t

    return lib, args, tunes


def _build_harness(args: list[Value], tunables: list[Value], omp_threads: int = 1) -> str:
    """
    Assemble the C harness file (from HARNESS_TEMPLATE).

    returns a string that can be written to a file and
    """
    assert all(isinstance(t.owner, TunableNode) for t in tunables)

    assert len(args) == 1
    assert args[0].type == FloatType(32)

    headers = ""
    if omp_threads > 1:
        headers += "#include <omp.h>"

    sigma_loops = "\n".join(
        f'{"    "*(i+1)}for (int step_{i} = 0; step_{i} < sigma_{i}_steps; step_{i}++) {{\n{"    "*(i+2)}float sweep_{i}_v = sigma_{i} + (sigma_{i}_max - sigma_{i}) * ((float) step_{i} / sigma_{i}_steps);\n'
        for i in range(len(tunables))
    )

    return HARNESS_TEMPLATE.format(
        headers=headers,
        omp_pragma=(
            f"#pragma omp parallel for num_threads({omp_threads})"
            if omp_threads > 1
            else ""
        ),
        omp_pragma_sweep=(
            f"#pragma omp parallel for num_threads({omp_threads})"
            if omp_threads > 1 and tunables
            else ""
        ),
        sigma_signature="".join(
            ", {} sigma_{}".format(type_to_llvm_type(t.type), i)
            for i, t in enumerate(tunables)
        ),
        sigma_signature_max="".join(
            ", {} sigma_{}_max".format(type_to_llvm_type(t.type), i)
            for i, t in enumerate(tunables)
        ),
        sigma_signature_steps="".join(
            ", int sigma_{}_steps".format(i) for i, _ in enumerate(tunables)
        ),
        sigma_count=len(tunables),
        sigma_args="".join(f", sigma_{i}" for i in range(len(tunables))),
        sigma_sweep_args="".join(f", sweep_{i}_v" for i in range(len(tunables))),
        sigma_loops=sigma_loops,
        sigma_loops_end="}" * len(tunables),
    )
