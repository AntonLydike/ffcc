import os
import sys
import shlex
import ctypes
import logging
import tempfile
import subprocess
from io import StringIO
from typing import Sequence

import numpy as np

from ffcc.platform import vector_width_bits
from ffcc.print_llvm import print_llvm_func_for, type_to_llvm_type
from ffcc.ir import IRNode, Value, TunableNode, VarNode, FloatType

LOGGER = logging.getLogger(__name__)

CLANG = os.environ.get("SYNTH_CLANG_BIN", "clang")

DEBUG_JIT_ENABLE = os.environ.get("FFCC_DEBUG_JIT", None)
"""
Allow debugging the jit
"""

HARNESS_TEMPLATE = """#include <stdint.h>
#include <math.h>
{headers}


typedef int32_t i32;
typedef int64_t i64;

{kernel_res_t} my_func({kernel_in_t} x{sigma_signature});
float my_func_scalar(float x{sigma_signature});

{helpers}

int eval_on_domain(float* restrict out, float* restrict domain, int64_t size{sigma_signature})
{{
    {omp_pragma}
    for (int64_t i = 0; i <= size - {vec_size}; i += {vec_size}) {{
        {vec_store} my_func({vec_load}{sigma_args}));
    }}

    // finish up remaining elements
    for (int64_t i =  size - (size % {vec_size}); i < size; i++) {{
        out[i] = my_func_scalar(domain[i]{sigma_args});
    }}

    return 0;
}}


int eval_on_linspace(float* restrict out, float low, float high, int64_t size{sigma_signature})
{{
    {omp_pragma}
    for (int64_t i = 0; i <= size - {vec_size}; i += {vec_size}) {{
        {vec_store} my_func(linspace_chunk(i, low, high, size){sigma_args}));
    }}
    
    // finish up remaining elements
    for (int64_t i =  size - (size % {vec_size}); i < size; i++) {{
        float frac = i / (float) size;
        out[i] = my_func_scalar(low * (1 - frac) + (high * frac){sigma_args});
    }}

    return 0;
}}


float max_relative_error(float* restrict reference, float* restrict domain, int64_t size, float epsilon{sigma_signature})
{{
    float max_rel_err = 0;
    
    int64_t chunk_size = {vec_size} * 1000;
    
    {omp_pragma}
    for (int64_t sec = 0; sec <= size - chunk_size; sec += chunk_size) {{
        
        float local_max = 0;
        
        for (int64_t i = sec; i < sec + chunk_size; i += {vec_size}) {{
            {vec_var} my_func({vec_load}{sigma_args}));

            local_max = {max_err_calc};
        }}
        
        {omp_critical}
        {{
            max_rel_err = fmax(max_rel_err, local_max);
        }}
    }}
    
    // finish up remaining elements
    for (int64_t i =  size - (size % chunk_size); i < size; i++) {{
        float res = my_func_scalar(domain[i]{sigma_args});
        max_rel_err = fmax(max_rel_err, fabsf(res - reference[i]) / (fabsf(reference[i]) + epsilon));
    }}

    return max_rel_err;
}}
"""

SSE_RELERR = """
// Compute max relative error of 4-wide float vectors using SSE
float _mm_relerr(__m128 res, __m128 ref, float eps) {
    __m128 eps_vec = _mm_set1_ps(eps);

    // |res - ref|
    __m128 diff = _mm_sub_ps(res, ref);
    __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);

    // |ref|
    __m128 abs_ref = _mm_andnot_ps(_mm_set1_ps(-0.0f), ref);

    // |ref| + eps
    __m128 denom = _mm_add_ps(abs_ref, eps_vec);

    // relerr = |res - ref| / (|ref| + eps)
    __m128 relerr = _mm_div_ps(abs_diff, denom);

    // Horizontal max reduction for 4 floats
    __m128 shuf = _mm_movehdup_ps(relerr);     // (1,1,3,3)
    __m128 max1 = _mm_max_ps(relerr, shuf);
    shuf = _mm_movehl_ps(shuf, max1);
    __m128 max2 = _mm_max_ss(max1, shuf);

    return _mm_cvtss_f32(max2);
}
"""

AVX2_RELERR = """
// Compute max relative error of 8-wide float vectors
float _mm256_relerr(__m256 res, __m256 ref, float eps) {
    __m256 eps_vec = _mm256_set1_ps(eps);

    // Absolute difference: |res - ref|
    __m256 diff = _mm256_sub_ps(res, ref);
    __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);

    // Absolute reference: |ref|
    __m256 abs_ref = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), ref);

    // Denominator: |ref| + eps
    __m256 denom = _mm256_add_ps(abs_ref, eps_vec);

    // Relative error: |res - ref| / (|ref| + eps)
    __m256 relerr = _mm256_div_ps(abs_diff, denom);

    // Horizontal max reduction of 8 floats
    __m128 low = _mm256_castps256_ps128(relerr);
    __m128 high = _mm256_extractf128_ps(relerr, 1);
    __m128 max1 = _mm_max_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(max1);     // duplicate high floats
    __m128 max2 = _mm_max_ps(max1, shuf);
    shuf = _mm_movehl_ps(shuf, max2);
    __m128 max3 = _mm_max_ss(max2, shuf);
    return _mm_cvtss_f32(max3);
}
"""


def linspace_code(typ: str, sfx: str, vec_size: int):
    if vec_size == 1:
        return "inline float linspace_chunk(int64_t i, float low, float high, int64_t size) {float f = i / (float) size;return low * (1 - f) + (high * f);}\n"
    indices = ", ".join(map(str, range(vec_size)))
    return f"""
inline {typ} linspace_chunk(int64_t i, float low, float high, int64_t size) {{
    // step 0: Compute local low, high and step:
    float frac = i / (float) size;
    float start = low * (1.0 - frac) + (high * frac);
    float step = (high - low) / size;
    
    // Step 1: Create vec of 0..{vec_size}
    {typ} indices = _mm{sfx}_setr_ps({indices});

    // Step 2: Broadcast constants
    {typ} scale = _mm{sfx}_set1_ps(step);
    {typ} start_vec = _mm{sfx}_set1_ps(start);

    // Step 3: Compute result: low + scale * (j)
    return _mm{sfx}_add_ps(start_vec, _mm{sfx}_mul_ps(scale, indices));
}}\n"""


AVX512_RELERR = """
float _mm512_relerr(__m512 res, __m512 ref, float eps) {
    __m512 eps_vec = _mm512_set1_ps(eps);

    // |res - ref|
    __m512 diff = _mm512_sub_ps(res, ref);
    __m512 abs_diff = _mm512_abs_ps(diff);  // AVX-512 has native abs!

    // |ref| + eps
    __m512 abs_ref = _mm512_abs_ps(ref);
    __m512 denom = _mm512_add_ps(abs_ref, eps_vec);

    // Relative error: |res - ref| / (|ref| + eps)
    __m512 relerr = _mm512_div_ps(abs_diff, denom);

    // Horizontal maximum of 16 floats
    float max_relerr = _mm512_reduce_max_ps(relerr);  // Native AVX-512 horizontal reduction

    return max_relerr;
}
"""


class _AUTO:
    pass


AUTOVEC = _AUTO()


class Program:
    tunables: list[TunableNode]
    variables: list[VarNode]
    dll: ctypes.CDLL

    def __init__(self, node: IRNode, num_threads: int = 4, vectorise: int | _AUTO = 1):
        if vectorise is AUTOVEC:
            vectorise = vector_width_bits() // node.type.width

        dll, args, tunes = instantiate_node_as_jit(
            node, omp_threads=num_threads, vectorise=vectorise
        )

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
        tunables: Sequence[float | int] | None = None,
        result: np.ndarray | None = None,
    ) -> np.ndarray:
        if result is None:
            result = np.zeros_like(domain)
        if tunables is None:
            tunables = self.initial_tune
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        if self.dll.eval_on_domain(result, domain, domain.size, *tunables) != 0:
            raise RuntimeError("evaulation failed")
        return result

    def eval_on_linspace(
        self,
        low: float,
        high: float,
        size: int,
        tunables: Sequence[float | int] = None,
        result: np.ndarray | None = None,
        endpoint: bool = True,
    ) -> np.ndarray:
        if result is None:
            result = np.zeros((size,), dtype=np.float32)
        if tunables is None:
            tunables = self.initial_tune
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        if endpoint:
            # put high one higher
            high = high + (high - low) / (size - 1)
        if self.dll.eval_on_linspace(result, low, high, size, *tunables) != 0:
            raise RuntimeError("evaulation failed")
        return result

    def max_relative_error(
        self,
        reference: np.ndarray,
        domain: np.ndarray,
        epsilon: float = 0.0,
        tunables: Sequence[float] = None,
    ) -> float:
        assert reference.size == domain.size
        if tunables is None:
            tunables = self.initial_tune
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        return self.dll.max_relative_error(
            reference, domain, domain.size, epsilon, *tunables
        )

    def __call__(self, x: float, *tunables: float | int) -> float:
        if len(tunables) == 0:
            tunables = self.initial_tune
        if len(tunables) != len(self.tunables):
            raise ValueError("Got the wrong number of tunables values")
        return self.dll.my_func_scalar(x, *tunables)


def instantiate_node_as_jit(
    node: IRNode,
    omp_threads: int = 4,
    lto: bool = True,
    opt_level: int = 3,
    vectorise: int = 1,
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
    args, tunes = None, None

    # do not re-generate files if jit debugging is enabled and files exist
    if not os.path.exists(kernel):
        with open(kernel, "w") as f:
            args, tunes = print_llvm_func_for(
                node, f, sym_name="my_func", vectorise=vectorise, add_scalar=True
            )
    else:
        args, tunes = print_llvm_func_for(
            node, StringIO(), sym_name="my_func", vectorise=vectorise
        )

    # FIXME: allow programs with signatures other than "x: f32"
    assert len(args) == 1
    input_t = args[0].type.ctype

    # do not re-generate files if jit debugging is enabled and files exist
    if not os.path.exists(harness):
        with open(harness, "w") as f:
            print(
                _build_harness(
                    args, tunes, omp_threads=omp_threads, vec_size=vectorise
                ),
                file=f,
            )

    cflags = ["-march=native", "-fuse-ld=lld", "-Wno-override-module"]
    if omp_threads > 1:
        cflags.append("-fopenmp")
    if lto:
        cflags.append("-flto=full")
    if opt_level > 0:
        cflags.append(f"-O{opt_level}")

    # make sure to read nixos added cflags
    if "NIX_CFLAGS_COMPILE_FOR_TARGET" in os.environ:
        cflags += shlex.split(os.environ["NIX_CFLAGS_COMPILE"])

    # make sure to read nixos added ldflags
    if "NIX_LDFLAGS_FOR_TARGET" in os.environ:
        cflags += shlex.split(os.environ["NIX_LDFLAGS_FOR_TARGET"])

    command = (CLANG, *cflags, "-lm", kernel, harness, "-shared", "-o", shared_obj)
    LOGGER.info(f"Compile command: {' '.join(command)}")

    if DEBUG_JIT_ENABLE is not None:
        input(
            f"JIT input files generated in {tmpdir}, press ENTER to continue compilation..."
        )

    try:
        subprocess.check_call(
            command,
            stderr=sys.stderr if LOGGER.level <= logging.INFO else subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        input(f"Look into {tmpdir}")
        raise e

    lib = ctypes.CDLL(shared_obj)

    # set types for eval_on_domain
    lib.eval_on_domain.argtypes = [
        np.ctypeslib.ndpointer(out_t),  # result
        np.ctypeslib.ndpointer(input_t),  # domain
        ctypes.c_int64,  # domain.size
        *(t.type.ctype for t in tunes),  # tunables
    ]

    lib.eval_on_domain.restype = ctypes.c_int

    lib.eval_on_linspace.argtypes = [
        np.ctypeslib.ndpointer(out_t),  # result
        ctypes.c_float,  # low
        ctypes.c_float,  # high
        ctypes.c_int64,  # domain.size
        *(t.type.ctype for t in tunes),  # tunables
    ]

    lib.eval_on_linspace.restype = ctypes.c_int

    # set types for max_relative_error
    lib.max_relative_error.argtypes = [
        np.ctypeslib.ndpointer(input_t),  # reference
        np.ctypeslib.ndpointer(input_t),  # domain
        ctypes.c_int64,  # domain.size = referece.size
        ctypes.c_float,  # epsilon
        *(t.type.ctype for t in tunes),  # tunables
    ]

    lib.max_relative_error.restype = ctypes.c_float

    # set input/output types
    # TODO: handle proper types here
    lib.my_func_scalar.argtypes = [input_t, *(t.type.ctype for t in tunes)]
    lib.my_func_scalar.restype = out_t

    return lib, args, tunes


def _build_harness(
    args: list[Value],
    tunables: list[Value],
    omp_threads: int = 1,
    vec_size: int = 1,
) -> str:
    """
    Assemble the C harness file (from HARNESS_TEMPLATE).

    returns a string that can be written to a file and
    """
    assert all(isinstance(t.owner, TunableNode) for t in tunables)

    assert len(args) == 1
    assert args[0].type == FloatType(32)

    headers = ""
    helpers = ""
    if omp_threads > 1:
        headers += "#include <omp.h>\n"

    sigma_args = "".join(f", sigma_{i}" for i in range(len(tunables)))
    sigma_signature = "".join(
        ", {} sigma_{}".format(type_to_llvm_type(t.type), i)
        for i, t in enumerate(tunables)
    )

    if vec_size > 1:
        width = vec_size * args[0].type.width
        # SSE are just _mm_ intrinsics
        if width == 128:
            suffix = ""
            type = "__m128"
            helpers = SSE_RELERR
        # avx/2 has _mm256_
        elif width == 256:
            suffix = "256"
            type = "__m256"
            helpers += AVX2_RELERR
        # and avx512 has _mm512_
        elif width == 512:
            suffix = "512"
            type = "__m512"
            helpers += AVX512_RELERR
        else:
            raise ValueError(f"Unsupported vector size: {width}")

        # add the helper for linspace
        helpers += linspace_code(type, suffix, vec_size)

        headers += "#include <immintrin.h>\n"

        vec_load = f"_mm{suffix}_loadu_ps(&domain[i])"
        vec_store = f"_mm{suffix}_storeu_ps(&out[i],"
        vec_var = f"{type} res = ("
        kernel_in_t = type
        kernel_res_t = type
        max_err_calc = f"fmax(local_max, _mm{suffix}_relerr(res, _mm{suffix}_loadu_ps(&reference[i]), epsilon))"
    else:
        # get scalar code
        helpers += linspace_code("", "", 1)
        vec_store = "out[i] = ("
        vec_load = "domain[i]"
        vec_var = f"float res = ("
        kernel_in_t = "float"
        kernel_res_t = "float"
        max_err_calc = "fmax(local_max, fabsf(res - reference[i]) / (fabsf(reference[i]) + epsilon))"
        helpers += f"\nfloat my_func_scalar(float x{sigma_signature}) {{return my_func(x{sigma_args});}}"

    return HARNESS_TEMPLATE.format(
        headers=headers,
        helpers=helpers,
        omp_pragma=(
            f"#pragma omp parallel for num_threads({omp_threads})"
            if omp_threads > 1
            else ""
        ),
        omp_critical="#pragma omp critical" if omp_threads > 1 else "",
        vec_size=vec_size,
        vec_store=vec_store,
        vec_load=vec_load,
        vec_var=vec_var,
        kernel_in_t=kernel_in_t,
        kernel_res_t=kernel_res_t,
        max_err_calc=max_err_calc,
        sigma_signature=sigma_signature,
        sigma_args=sigma_args,
    )
