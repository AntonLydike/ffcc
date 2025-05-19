# RUN: python %s | filecheck --match-full-lines %s
import math

from ffcc.cse import cse
from ffcc.ir import VarNode, ConstantNode, FloatType, IRNode, MathNode, Kind
from ffcc.diff import diff
from ffcc.print import print_dag
from ffcc.opt import simp

def show(dag: IRNode):
    print(print_dag(simp(cse(dag))))

f32 = FloatType(32)
c0 = ConstantNode(0, f32)
c1 = ConstantNode(1, f32)
c2 = ConstantNode(2, f32)
e = ConstantNode(math.e, f32)
half = ConstantNode(1/2, f32)
x = VarNode('x', f32)

show(diff(x, x))
# CHECK: 1

show(diff(c0, x))
# CHECK: 0

show(simp(diff(x**c1, x)))
# CHECK: 1

show(simp(diff(x**c2, x)))
# CHECK: mul(2, x)

show(simp(diff(x*x, x)))
# CHECK: mul(2, x)

show(simp(diff(x**c2 + x, x)))
# CHECK: add(1, mul(2, x))

show(simp(diff(x**(-half), x)))
# CHECK: mul(-0.5, pow(x, -1.5))

show(diff(MathNode(x, kind=Kind.Floor, res_type=f32), x))
# CHECK: 0

show(diff((x*x)/c2, x))
# CHECK: x

show(diff(c2**(x**c2), x))
# 2 * ln(2) = 1.3862943611198906
# CHECK: mul(1.3862943611198906, mul(x, pow(2, pow(x, 2))))

show(diff(x / (c1 + e ** (-x)), x))
