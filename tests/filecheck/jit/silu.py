# RUN: python %s | filecheck %s
import sys

from ffcc.parse import parse_ssa
from ffcc.jit import Program
from ffcc.tune import GreedyDescent
from ffcc.helper import duration

import numpy as np


num_elems = 100_000

x = np.linspace(-4, 4, num_elems, dtype=np.float32)
# compute reference silu:
expected =  x / (1 + np.exp(-x))


#p = Program(node := parse_ssa("""%sigma = tunable 's2' = 2129950675 : i32
#%one = tunable 'one' = 1 : f32
#%sigma1 = tunable 's1' = 1064975338 : i32
#%0 = tunable 'L' = 12102203.161561485 : f32
#%x = var 'x' : f32
#%mx = negate %x : f32
#%1 = mul %0, %mx : f32
#%2 = add %sigma1, %1 : i32
#%3 = bitcast i2f %2 to f32
#%one_p_ex = add %one, %3 : f32
#%4 = bitcast f2i %one_p_ex to i32
#%5 = sub %sigma, %4 : i32
#%6 = bitcast i2f %5 to f32
#%r = mul %x, %6 : f32
#"""), num_threads=4)

p = Program(node := parse_ssa("""%x = var 'x' : f32
%sigma = tunable 's2' = 2129881071 : i32
%one = tunable 'one' = 1.0119495391845703 : f32
%sigma1 = tunable 's1' = 1065039594 : i32
%0 = tunable 'L' = 12000943.0 : f32
%mx = negate %x : f32
%1 = mul %0, %mx : f32
%2 = add %sigma1, %1 : i32
%3 = bitcast i2f %2 to f32
%one_p_ex = add %one, %3 : f32
%4 = bitcast f2i %one_p_ex to i32
%5 = sub %sigma, %4 : i32
%y = bitcast i2f %5 to f32

// x * y * y
%_0 = mul %one_p_ex, %y : f32
%_1 = mul %_0, %y : f32

// 2 * y
%c2 = tunable two = 2 : f32
%_2 = mul %c2, %y : f32
%yn = sub %_2, %_1 : f32
%r = mul %x, %yn : f32
"""))

print("Tunables for silu:")
# CHECK-LABEL: Tunables for silu:
for tune in p.tunables:
    print("Tune {} = {} : {}".format(tune.name, tune.hint, tune.type))
# CHECK-DAG: Tune L = {{-?[\d\.]+}} : f32
# CHECK-DAG: Tune one = {{-?[\d\.]+}} : f32
# CHECK-DAG: Tune s2 = {{-?\d+}} : i32
# CHECK-DAG: Tune s1 = {{-?\d+}} : i32


# get initial values for tuning
initial_tune = p.initial_tune

print("max rel error with initial tune (ε=0): {}".format(p.max_relative_error(expected, x, 0.0, initial_tune)))
print("max rel error with initial tune (ε=0.1): {}".format(p.max_relative_error(expected, x, 0.5, initial_tune)))
print("max rel error with initial tune (ε=1): {}".format(p.max_relative_error(expected, x, 1.0, initial_tune)))
# CHECK: max rel error with initial tune (ε=0): 0.053504280745983124
# CHECK: max rel error with initial tune (ε=0.1): 0.013524065725505352
# CHECK: max rel error with initial tune (ε=1): 0.011155398562550545

### GREEDY DESCENT:

print("Greedy descending...")
# CHECK-LABEL: Greedy descending
desc = GreedyDescent(p, x, expected, 128, 512, 1, 20, epsilon=1.0)
greedy_tune = desc.run(progress=sys.stdout.isatty())

print(f"best found tune: {greedy_tune}")
# CHECK: best found tune: (12011183.0, 2.00244140625, 1.0107288360595703, 2129883631, 1065037034)
print("max rel err: {}".format(p.max_relative_error(expected, x, 1.0, greedy_tune)))
# CHECK: max rel err: 0.008350450545549393

print_ir = sys.stdout.isatty()
# print the IR
if print_ir:
    from ffcc.print import print_ssa
    for tune, new_val in zip(p.tunables, greedy_tune):
        tune.hint = new_val
    print_ssa(node)
