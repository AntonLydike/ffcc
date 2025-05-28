# RUN: python %s

import numpy as np
from ffcc.jit import Program
from ffcc.parse import parse_ssa
from ffcc.platform import has_comp_flag

node = parse_ssa("""%x = var 'x' : f32
%sigma = tunable 'sigma' = 2129950675 : i32
%one = constant 1 : f32
%sigma1 = tunable 'sigma' = 1064975338 : i32
%0 = constant 12102203.161561485 : f32
%mx = negate %x : f32
%1 = mul %0, %mx : f32
%2 = add %sigma1, %1 : i32
%3 = bitcast i2f %2 to f32
%one_p_ex = add %one, %3 : f32
%4 = bitcast f2i %one_p_ex to i32
%5 = sub %sigma, %4 : i32
%6 = bitcast i2f %5 to f32
""")


scalar = Program(node, vectorise=1)
sse = Program(node, vectorise=4)

domain = np.linspace(-6, 6, 100_000, dtype=np.float32)

ref = scalar.eval_on_domain(domain)

assert np.all(np.isclose(
    ref,
    sse.eval_on_domain(domain),
))
print("SSE pass")


if has_comp_flag('avx2'):
    avx2 = Program(node, vectorise=8)
    assert np.all(np.isclose(
        ref,
        avx2.eval_on_domain(domain),
    ))
    print("AVX2 pass")
else:
    print("AVX2 skipped")

if has_comp_flag('avx512'):
    avx512 = Program(node, vectorise=16)
    assert np.all(np.isclose(
        ref,
        avx512.eval_on_domain(domain),
    ))
    print("AVX512 pass")
else:
    print("AVX512 skipped")
