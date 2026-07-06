import itertools
import math
import sys
import time
from typing import Any, Callable, Literal, cast

from aalib.colors import FMT
import torch
import torch.nn as nn
from aalib.progress import progress, simple_progress
from torch import nn, tensor

from ffcc.eval import evaluate
from ffcc.ir import (
    BitCastOperator,
    ConstantNode,
    FloatType,
    IntType,
    IRNode,
    Kind,
    MathNode,
    TunableNode,
    Type,
    Value,
    VarNode,
)
from ffcc.parse import Expression

LOGOPS = {
    math.e: torch.log,
    2: torch.log2,
    10: torch.log10,
}


def to_torch_type(t: Type):
    match t:
        case IntType(w):
            if hasattr(torch, f"int{w}"):
                return getattr(torch, f"int{w}")
            raise ValueError(f"Torch does not support int{w}")
        case FloatType(w):
            if hasattr(torch, f"float{w}"):
                return getattr(torch, f"float{w}")
            raise ValueError(f"Torch does not support float{w}")
        case v:
            raise ValueError(v)


class TunableIRModule(nn.Module):
    def __init__(
        self,
        vars: tuple[VarNode, ...],
        ir: IRNode,
    ):
        super().__init__()
        self.ir = ir
        self.vars = tuple(v.result for v in vars)
        tunables: list[TunableNode] = []
        for node in ir.walk():
            if isinstance(node, TunableNode) and node not in tunables:
                tunables.append(node)
        self.tunables: dict[Value, nn.Parameter] = {
            t.result: nn.Parameter(tensor([t.hint], dtype=to_torch_type(t.type)))
            for t in tunables
        }
        self._casts = {}
        self._params = nn.ParameterList(self.tunables.values())

    def real_params(self) -> list[float]:
        return [e.item() for e in self.tunables.values()]

    def assign_back(self):
        """assign the trained parameters back to the tunables of the expression"""
        for tunable, param in self.tunables.items():
            tunable.owner.hint = param.data.item()

    def forward(self, *vals: tensor) -> tensor:
        var_to_val: dict[Value, torch.tensor | nn.Parameter | float] = dict(
            itertools.chain(zip(self.vars, vals), self.tunables.items())
        )
        for node in self.ir.walk(reverse=True):
            args = [var_to_val[a] for a in node.args]
            match node:
                case TunableNode() | VarNode():
                    pass
                case ConstantNode(val):
                    var_to_val[node.result] = val
                case MathNode(
                    kind=Kind.Add | Kind.Sub | Kind.Mul | Kind.Div | Kind.Negate
                ):
                    var_to_val[node.result] = node.evaluate(args)
                case MathNode(kind=Kind.Log, argops=(_, ConstantNode(base))):
                    if base in LOGOPS:
                        res = LOGOPS[base](args[0])
                    else:
                        # change of basis
                        res = torch.log(args[0]) / torch.log(base)
                    var_to_val[node.result] = res
                case MathNode(kind=Kind.Pow):
                    var_to_val[node.result] = torch.pow(*args)
                case MathNode(k):
                    raise ValueError("Unsupported math op kind", k)
                case BitCastOperator(direction):
                    res = self._do_bitcast(args[0], direction, node.result.type)
                    var_to_val[node.result] = res
                case _:
                    raise ValueError("Unsupported operator", node)
        # return final result
        return var_to_val[self.ir.result]

    def _do_bitcast(
        self, arg: tensor, direction: Literal["i2f", "f2i"], dest_t: Type
    ) -> tensor:
        src_t = IntType(dest_t.width) if direction == "i2f" else FloatType(dest_t.width)
        if (src_t, dest_t) not in self._casts:
            self._casts[(src_t, dest_t)] = make_bc(
                to_torch_type(src_t), to_torch_type(dest_t)
            )
        return self._casts[(src_t, dest_t)](arg)


def tune(
    base_exp: Expression,
    approximation: Expression,
    domain: tuple[float, float],
    samples: int = 100_000,
) -> float:
    """
    Takes a base expression and an approximation with tunable variables, and changes the
    tunables to a local minima found by doing gradient descent using pytorch.

    Returns the mean-squared-error of the final expression.
    """
    dtype = to_torch_type(base_exp.expr.result.type)
    domain_t = torch.linspace(*domain, samples, dtype=dtype)

    model = TunableIRModule(
        approximation.variables,
        approximation.expr,
    )

    criterion = nn.MSELoss()
    baseline = tensor(
        evaluate(
            base_exp.expr, {v.result: domain_t.numpy() for v in base_exp.variables}
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)

    epochs = 10000
    t0 = time.time()
    loss = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(domain_t), baseline)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch) % 100 == 0 and sys.stderr.isatty():
            simple_progress(
                epoch + 1,
                epochs,
                t0,
                f"loss={loss:.8f}, lr={scheduler.get_last_lr()[0]:8.4f}",
                color=FMT.ORANGE,
                file=sys.stderr,
            )
    # assign trained params back to tunable params:
    model.assign_back()
    print()
    return loss


# --------------------------------
# Backwards Helper:
# --------------------------------


def with_backwards(gradients: Callable[[torch.Tensor], torch.Tensor]):
    def wrapper(
        fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                ctx.save_for_backward(x)
                return fn(x)

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                (x,) = ctx.saved_tensors
                return gradients(x) * grad_output

        return CustomFunction.apply

    return wrapper


def make_bc(src_t, dest_t) -> Callable[[torch.Tensor], torch.Tensor]:
    bitwidth = dest_t.itemsize * 8
    assert src_t.itemsize == dest_t.itemsize, "source and dest type must have same size"
    assert bitwidth in (16, 32, 64), "special bitwidths are unsupported"
    L = {
        64: 2**52,
        32: 2**23,
        16: 2**10,
    }[bitwidth]
    B = {
        64: 1023,
        32: 127,
        16: 15,
    }[bitwidth]

    class BitCast(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> Any:
            ctx.save_for_backward(x)
            return x.type(src_t).view(dest_t)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            if dest_t.is_floating_point:
                x = ctx.saved_tensors[0]
                return grad_output * torch.pow(2, torch.floor(x / L) - B) / L
            else:
                raise NotImplementedError()

    return BitCast.apply
