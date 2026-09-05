import itertools
from logging import getLogger
import math
import sys
import time
from typing import Any, Callable, Literal

import torch
from aalib.colors import Color
from aalib.progress import simple_progress
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

LOGGER = getLogger(__name__)

LOGOPS = {
    math.e: torch.log,
    2: torch.log2,
    10: torch.log10,
}

# (width) -> (mantissa bits, bias)
_IEEE = {16: (10, 15), 32: (23, 127), 64: (52, 1023)}


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
        # Parameters are kept in at least f32: f16 cannot represent Adam's
        # epsilon (1e-8) or typical gradients, so optimization would NaN or
        # stall. The tunables are O(1) (the approximate pass guarantees it),
        # so the f32 -> f16 cast in the forward does not lose expressiveness.
        self.tunables: dict[Value, nn.Parameter] = {}
        for t in tunables:
            work_dtype = to_torch_type(t.type)
            param_dtype = (
                torch.float32 if t.type.width < 32 else work_dtype
            )
            self.tunables[t.result] = nn.Parameter(
                tensor([t.hint], dtype=param_dtype)
            )
        self.initial_params = tuple(t.item() for t in self.tunables.values())
        self._casts = {}
        self._params = nn.ParameterList(self.tunables.values())

    def param_values(self) -> tuple[float, ...]:
        return tuple(e.item() for e in self.tunables.values())

    def freeze(self):
        """Replace the trained tunables with plain constants of their final values."""
        for value, param in self.tunables.items():
            value.replace_with(
                ConstantNode(param.data.item(), value.owner.type).result
            )

    def forward(self, *vals: torch.Tensor):
        return self._eval(vals, to_torch_type(self.ir.result.type))

    def _eval(self, vals, dtype):
        """Evaluate the IR with all arithmetic in `dtype`.

        Tunables are read from the parameters and cast to `dtype`, so the
        graph stays differentiable through them.
        """
        # a single domain tensor is broadcast to every variable
        if len(vals) == 1 and len(self.vars) > 1:
            vals = vals * len(self.vars)
        var_to_val: dict[Value, torch.Tensor | nn.Parameter | float] = dict(
            itertools.chain(
                zip(self.vars, vals),
                ((r, p.to(dtype)) for r, p in self.tunables.items()),
            )
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
                    res = self._do_bitcast(args[0], direction, dtype)
                    var_to_val[node.result] = res
                case _:
                    raise ValueError("Unsupported operator", node)
        # return final result
        return var_to_val[self.ir.result]

    def _do_bitcast(
        self,
        arg: torch.Tensor,
        direction: Literal["i2f", "f2i"],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if direction == "i2f":
            src_t, dest_t = to_torch_type(IntType(dtype.itemsize * 8)), dtype
        else:
            src_t, dest_t = dtype, to_torch_type(IntType(dtype.itemsize * 8))
        if (src_t, dest_t) not in self._casts:
            self._casts[(src_t, dest_t)] = make_bc(src_t, dest_t)
        return self._casts[(src_t, dest_t)](arg)


def tune(
    base_exp: Expression,
    approximation: Expression,
    domain: tuple[float, float],
    samples: int = 4096,
    lr: float = 1e-3,
    record: list | None = None,
):
    """
    Tune the tunables of `approximation` to minimize the MSE against
    `base_exp` over `domain` with Adam, then freeze them into plain
    constants, so `approximation` is left with no tunables. The parameters
    of the lowest-loss epoch are kept, since the loss is quantized for
    sub-f32 targets and the final epoch can sit on a worse rounding pattern
    than an earlier one.
    """
    dtype = to_torch_type(base_exp.expr.result.type)

    domain_t = torch.linspace(*domain, samples, dtype=dtype)
    model = TunableIRModule(approximation.variables, approximation.expr)
    criterion = nn.MSELoss()
    baseline = tensor(
        evaluate(
            base_exp.expr, {v.result: domain_t.numpy() for v in base_exp.variables}
        )
    ).to(dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 600
    t0 = time.time()
    with torch.no_grad():
        initial_loss = float(criterion(model._eval((domain_t,), dtype), baseline))
    best_loss = initial_loss
    best_params = [p.detach().clone() for p in model.parameters()]
    if record is not None:
        record.append(initial_loss)

    def progress():
        if sys.stderr.isatty():
            simple_progress(
                epoch + 1,
                epochs,
                t0,
                f"loss={loss:.8f}, lr={lr:.1e}",
                color=Color.YELLOW,
                file=sys.stderr,
            )

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model._eval((domain_t,), dtype)
        loss = criterion(out, baseline)
        loss.backward()
        optimizer.step()
        loss = loss.detach().item()
        if loss < best_loss:
            best_loss = loss
            best_params = [p.detach().clone() for p in model.parameters()]
        if record is not None:
            record.append(loss)
        if epoch % 10 == 9:
            progress()
    if sys.stderr.isatty():
        print(file=sys.stderr)
    # keep the parameters of the lowest-loss epoch
    with torch.no_grad():
        for p, v in zip(model.parameters(), best_params):
            p.copy_(v)
    LOGGER.info(
        f"Tuned parameters {model.initial_params} -> {model.param_values()}, "
        f"improving MSE from {initial_loss:.8f} to {best_loss:.8f}"
    )
    model.freeze()


# --------------------------------
# Backwards Helper:
# --------------------------------


def make_bc(src_t, dest_t) -> Callable[[torch.Tensor], torch.Tensor]:
    bitwidth = dest_t.itemsize * 8
    assert src_t.itemsize == dest_t.itemsize, "source and dest type must have same size"
    assert bitwidth in (16, 32, 64), "special bitwidths are unsupported"
    L, B = _IEEE[bitwidth]
    L = 2**L
    B = B

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
