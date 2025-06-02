# RUN: python %s
import numpy as np

from ffcc.jit import Program, AUTOVEC
from ffcc.ir import VarNode, FloatType

f32 = FloatType(32)

incr = Program(VarNode('x', f32) + 1.0, num_threads=1)
id = Program(VarNode('x', f32), num_threads=1)
mul = Program(VarNode('x', f32) * 2.0, num_threads=1)

num_elems = 10

domain = np.linspace(-5, 5, num_elems, dtype=np.float32)
domain_open = np.linspace(-5, 5, num_elems, endpoint=False, dtype=np.float32)

def check_isclose(reference: np.ndarray, b: np.ndarray):
    for i, (ref, e1) in enumerate(zip(reference, b)):
        if not np.isclose(e1, ref):
            print(f"a = {reference}\nb = {b}")
            print(f"Mismatching values at index {i}: {ref} != {e1}")
            return False
    return True

print("#### id:")
assert check_isclose(domain, id.eval_on_domain(domain))
print("eval_on_domain: check")
assert check_isclose(domain, id.eval_on_linspace(-5, 5, num_elems, endpoint=True))
print("eval_on_linspace(endpoint=True): check")
assert check_isclose(domain_open, id.eval_on_linspace(-5, 5, num_elems, endpoint=False))
print("eval_on_linspace(endpoint=False): check")

print("#### incr:")
assert check_isclose(domain + 1, incr.eval_on_domain(domain))
print("eval_on_domain: check")
assert check_isclose(domain + 1, incr.eval_on_linspace(-5, 5, num_elems, endpoint=True))
print("eval_on_linspace(endpoint=True): check")
assert check_isclose(domain_open + 1, incr.eval_on_linspace(-5, 5, num_elems, endpoint=False))
print("eval_on_linspace(endpoint=False): check")

print("#### mul:")
assert check_isclose(domain * 2, mul.eval_on_domain(domain))
print("eval_on_domain: check")
assert check_isclose(domain * 2, mul.eval_on_linspace(-5, 5, num_elems, endpoint=True))
print("eval_on_linspace(endpoint=True): check")
assert check_isclose(domain_open * 2, mul.eval_on_linspace(-5, 5, num_elems, endpoint=False))
print("eval_on_linspace(endpoint=False): check")


print("#### vectorized incr:")

incr_vec = Program(VarNode('x', f32) + 1.0, num_threads=1, vectorise=AUTOVEC)

num_elems = 10_000
domain = np.linspace(-5, 5, num_elems, dtype=np.float32)

domain_open = np.linspace(-5, 5, num_elems, dtype=np.float32, endpoint=False)

import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use("TkAgg")
#def plot(n, r):
#    r[r == 0] = np.nan
#    plt.scatter(
#        domain,
#        r,
#        alpha=0.3,
#        label=n,
#        s=1,
#    )
#
#plot('incr', np.abs(domain + 1 - incr.eval_on_linspace(-5, 5, num_elems, endpoint=True)) + 0e-9)
#plot('incr_vec', np.abs(domain + 1 - incr_vec.eval_on_linspace(-5, 5, num_elems, endpoint=True)) + 5e-9)
#plot('incr(end=0)', np.abs(domain_open + 1 - incr.eval_on_linspace(-5, 5, num_elems, endpoint=False)) + 10e-9)
#plot('incr_vec(end=0)', np.abs(domain_open + 1 - incr_vec.eval_on_linspace(-5, 5, num_elems, endpoint=False)) + 15e-9)
#
#plt.title("absolute error of (x+1) on domain -5,5 - MODE MUL FIRST")
#for lh in plt.legend().legend_handles:
#    lh.set_alpha(1)
#plt.show()

assert check_isclose(domain + 1, incr.eval_on_domain(domain))
print("eval_on_domain: check")
assert check_isclose(domain + 1, incr.eval_on_linspace(-5, 5, num_elems, endpoint=True))
print("eval_on_linspace(endpoint=True): check")
assert check_isclose(domain_open + 1, incr.eval_on_linspace(-5, 5, num_elems, endpoint=False))
print("eval_on_linspace(endpoint=False): check")

