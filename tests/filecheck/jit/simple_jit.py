# RUN: python %s | filecheck %s
import numpy as np
from ffcc.ir import FloatType, MathNode, ConstantNode, Kind, VarNode, TunableNode
from ffcc.jit import Program

f32 = FloatType(32)

x = VarNode('x', f32)
t = TunableNode('t', 1, f32)
node = x + (t * ConstantNode(3.14159, f32))

p = Program(node)

domain = np.linspace(1, 2, 100, dtype=np.float32)
sigmas = np.array([1], dtype=np.float32)

# individual call
print(p.dll.my_func_scalar.argtypes)
# CHECK: [<class 'ctypes.c_float'>, <class 'ctypes.c_float'>]
print(
    'my_func(0, 1) = {}'.format(p(0, 1))
)
# CHECK-NEXT: my_func(0, 1) = 3.141590118408203

# evaluate on [-1, 1]
results = p.eval_on_domain(domain, (1,))
# CHECK-NEXT: eval_on_domain [1, 2] -> [4.14159   4.151691  4.1617923 4.171893  4.1819944 4.1920953 4.202196
# CHECK-NEXT:  4.2122974 4.2223983 4.232499  4.2426004 4.2527013 4.262802  4.2729034
# CHECK-NEXT:  4.2830043 4.293105  4.3032064 4.3133073 4.323408  4.3335094 4.3436103
# CHECK-NEXT:  4.353711  4.3638124 4.3739133 4.384014  4.3941154 4.4042163 4.414317
# CHECK-NEXT:  4.4244184 4.4345193 4.44462   4.4547215 4.4648223 4.4749236 4.4850245
# CHECK-NEXT:  4.4951253 4.5052266 4.5153275 4.525429  4.5355296 4.5456305 4.555732
# CHECK-NEXT:  4.5658326 4.5759335 4.586035  4.5961356 4.6062365 4.616338  4.6264386
# CHECK-NEXT:  4.6365395 4.646641  4.6567416 4.6668425 4.676944  4.6870446 4.6971455
# CHECK-NEXT:  4.707247  4.7173476 4.7274485 4.73755   4.7476506 4.7577515 4.767853
# CHECK-NEXT:  4.7779536 4.788055  4.798156  4.8082566 4.818358  4.828459  4.83856
# CHECK-NEXT:  4.848661  4.858762  4.868863  4.878964  4.889065  4.899166  4.909267
# CHECK-NEXT:  4.919368  4.929469  4.93957   4.949671  4.959772  4.969873  4.979974
# CHECK-NEXT:  4.990075  5.000176  5.010277  5.020378  5.030479  5.04058   5.050681
# CHECK-NEXT:  5.060782  5.070883  5.080984  5.091085  5.101186  5.111287  5.121388
# CHECK-NEXT:  5.1314893 5.14159  ]


print('eval_on_domain [1, 2] -> {}'.format(results))

# relative error:
actual_results = np.ones_like(domain)*2
print(
    'max_rel_err [1, 2] = {:.8f}'.format(p.max_relative_error(actual_results, domain, tunables=(0.1, )))
)
# CHECK-NEXT: max_rel_err [1, 2] = 0.34292048

print(
    'max_rel_err [1, 2], eps = 1 = {:.8f}'.format(p.max_relative_error(actual_results, domain, tunables=(0.1, ), epsilon=1))
)
# CHECK-NEXT: max_rel_err [1, 2], eps = 1 = 0.22861366