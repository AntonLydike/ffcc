from ffcc.ir import VarNode, FloatType
from ffcc.jit import Program, AUTOVEC
from ffcc.print import print_ssa
x = VarNode('x',FloatType(32))

import numpy as np

ir = x ** -0.5

print_ssa(ir)

p = Program(ir, num_threads=4, vectorise=AUTOVEC)

print(
    p.eval_on_domain(np.linspace(1, 4, 100_000))
)
