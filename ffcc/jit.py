import sys

from typing import Sequence

import os
import shlex
import ctypes
import tempfile
import subprocess

import numpy as np

from ffcc.print_llvm import print_llvm_func_for, type_to_llvm_type

from ffcc.ir import IRNode, Value, TunableNode, VarNode

CLANG = os.environ.get("SYNTH_CLANG_BIN", "clang")

HARNESS_TEMPLATE = """#include <math.h>
#include <stdint.h>
{headers}

float my_func(float x, {sigma_signature});

int eval_on_domain(float* restrict out, float* restrict domain, int64_t size, {sigma_signature})
{{

    {omp_pragma}
    for (int64_t i = 0; i < size; i++) {{
        out[i] = my_func(domain[i], {sigma_args});
    }}
    return 0;
}}

float max_relative_error(float* restrict reference, float* restrict domain, int64_t size, {sigma_signature})
{{
    float max_rel_err = -INFINITY;

    {omp_pragma}
    for (int64_t i = 0; i < size; i++) {{
        float res = my_func(domain[i], {sigma_args});
        float rel_err = fabsf((res - reference[i]) / reference[i]);
        max_rel_err = fmax(max_rel_err, rel_err);
    }}

    return max_rel_err;
}}

int64_t sweep_tunables(float* restrict output, float* restrict reference, float* restrict domain, int64_t size, {sigma_signature}, {sigma_signature_max}, {sigma_signature_steps})
{{
    int64_t res_idx = 0;
    
    {omp_pragma}
{sigma_loops}

        output[res_idx++] = max_relative_error(reference, domain, size, {sigma_sweep_args});
    
    {sigma_loops_end}

    return res_idx;
}}
"""


class Program:
    tunables: list[TunableNode]
    variables: list[VarNode]
    dll: ctypes.CDLL

    def __init__(self, node: IRNode):
        dll, args = instantiate_node_as_jit(node)

        self.variables = []
        self.dll = dll
        # first take all var nodes:
        while isinstance(args[0].owner, VarNode):
            self.variables.append(args.pop(0).owner)
        # all remaining nodes are tunables
        self.tunables = [arg.owner for arg in args]

        assert len(self.variables) == 1, "Cannot handle multi-variable programs yet!"

    @property
    def initial_tune(self) -> tuple[float, ...]:
        return tuple(t.hint for t in self.tunables)

    def eval_on_domain(
        self,
        domain: np.ndarray,
        tunables: Sequence[float],
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

    def __call__(self, x: float, *tunables: float) -> float:
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        return self.dll.my_func(x, *tunables)


def instantiate_node_as_jit(
    node: IRNode, omp_threads: int = 4, lto: bool = True, opt_level: int = 3
) -> tuple[ctypes.CDLL, list[Value]]:
    with tempfile.TemporaryDirectory() as tmpdir:

        kernel = os.path.join(tmpdir, "kernel.ll")
        harness = os.path.join(tmpdir, "harness.c")
        shared_obj = os.path.join(tmpdir, "out.so")

        with open(kernel, "w") as f:
            args = print_llvm_func_for(node, "my_func", f)
        len_tunes = len(args) - 1

        with open(harness, "w") as f:
            print(_build_harness(args, omp_threads=omp_threads), file=f)

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

        subprocess.check_call(
            [CLANG, *cflags, "-lm", kernel, harness, "-shared", "-o", shared_obj],
            # stderr=subprocess.DEVNULL
        )

        lib = ctypes.CDLL(shared_obj)

        # set types for eval_on_domain
        lib.eval_on_domain.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float),  # result
            np.ctypeslib.ndpointer(ctypes.c_float),  # domain
            ctypes.c_int64,  # domain.size
        ] + [
            ctypes.c_float for _ in range(len_tunes)
        ]  # *sigma

        lib.eval_on_domain.restype = ctypes.c_int

        # set types for max_relative_error
        lib.max_relative_error.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_float),  # reference
            np.ctypeslib.ndpointer(ctypes.c_float),  # domain
            ctypes.c_int64,  # domain.size = referece.size
        ] + [
            ctypes.c_float for _ in range(len_tunes)
        ]  # *sigma

        lib.max_relative_error.restype = ctypes.c_float

        # set types for sweep_tunables
        lib.sweep_tunables.argtypes = (
            [
                np.ctypeslib.ndpointer(ctypes.c_float),  # output of sweep
                np.ctypeslib.ndpointer(ctypes.c_float),  # reference
                np.ctypeslib.ndpointer(ctypes.c_float),  # domain
                ctypes.c_int64,  # domain.size = referece.size
            ]
            + [ctypes.c_float for _ in range(len_tunes * 2)]
            + [ctypes.c_int for _ in range(len_tunes)]
        )

        lib.sweep_tunables.restype = ctypes.c_int64

        # set input/output types
        # TODO: handle proper types here
        lib.my_func.argtypes = [ctypes.c_float] * len(args)
        lib.my_func.restype = ctypes.c_float

        return lib, args


def _build_harness(args: list[Value], omp_threads: int = 1) -> str:
    """
    Assemble the C harness file (from HARNESS_TEMPLATE).

    returns a string that can be written to a file and
    """
    args, tunables = args[:1], args[1:]
    assert all(isinstance(t.owner, TunableNode) for t in tunables)

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
        sigma_signature=", ".join(
            "{} sigma_{}".format(type_to_llvm_type(t.type), i)
            for i, t in enumerate(tunables)
        ),
        sigma_signature_max=", ".join(
            "{} sigma_{}_max".format(type_to_llvm_type(t.type), i)
            for i, t in enumerate(tunables)
        ),
        sigma_signature_steps=", ".join(
            "int sigma_{}_steps".format(i) for i, _ in enumerate(tunables)
        ),
        sigma_count=len(tunables),
        sigma_args=", ".join(f"sigma_{i}" for i in range(len(tunables))),
        sigma_sweep_args=", ".join(f"sweep_{i}_v" for i in range(len(tunables))),
        sigma_loops=sigma_loops,
        sigma_loops_end="}" * len(tunables),
    )
