## RUN: python %s | filecheck %s

from ffcc.parse import parse_expr

print(parse_expr("test1(x) = x"))
## CHECK: Expression(name='test1', variables=(x), expr=x)

print(parse_expr("test2(x) = ln(x)"))
## CHECK: name='test2'
## CHECK-SAME: variables=(x)
## CHECK-SAME: expr=ln(x)

print(parse_expr("test3(y) = exp(y)"))
## CHECK: name='test3'
## CHECK-SAME: variables=(y)
## CHECK-SAME: expr=pow(e, y)

print(parse_expr("precedence(x) = 1 + x / 2"))
## CHECK: name='precedence'
## CHECK-SAME: variables=(x)
## CHECK-SAME: expr=add(1.0, div(x, 2.0))

print(parse_expr("precedence2(x) = 1 * 2 + x"))
## CHECK: name='precedence2'
## CHECK-SAME: variables=(x)
## CHECK-SAME: expr=add(mul(1.0, 2.0), x)

print(parse_expr("precedence3(x) = 1 * 2 * 3"))
## CHECK: name='precedence3'
## CHECK-SAME: variables=(x)
## CHECK-SAME: expr=mul(mul(1.0, 2.0), 3.0)
