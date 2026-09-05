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
        self._surr_casts: dict[int, tuple[Callable, Callable]] = {}
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

    def _eval(self, vals, dtype, surrogate: int | None = None):
        """Evaluate the IR with all arithmetic in `dtype`.

        Tunables are read from the parameters and cast to `dtype`, so the
        graph stays differentiable through them. If `surrogate` is given
        (a bitwidth < 32), bitcasts implement that bitwidth's semantics on
        f32 tensors with f32-exact gradients (see `surrogate_bc`); this is
        used to take gradients of sub-f32 graphs, where the native f16
        backward underflows to zero.
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
                    res = self._do_bitcast(args[0], direction, dtype, surrogate)
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
        surrogate: int | None = None,
    ) -> torch.Tensor:
        if surrogate is not None:
            if surrogate not in self._surr_casts:
                self._surr_casts[surrogate] = surrogate_bc(surrogate)
            i2f, f2i = self._surr_casts[surrogate]
            return i2f(arg) if direction == "i2f" else f2i(arg)
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
    samples: int = int(1e5),
    lr: float = 1e-3,
    record: list | None = None,
):
    """
    Tune the tunables of `approximation` to minimize the MSE against
    `base_exp` over `domain` with Adam, then freeze them into plain
    constants, so `approximation` is left with no tunables. The parameters
    of the lowest-loss epoch (measured in native precision) are kept, since
    the reported loss is quantized for sub-f32 targets and the final epoch
    can sit on a worse rounding pattern than an earlier one.
    """
    width = base_exp.expr.result.type.width
    dtype = to_torch_type(base_exp.expr.result.type)
    # For sub-f32 widths the native backward underflows (the analytic
    # gradient through the bitcast falls below the subnormal floor and zeros
    # out), so gradients are taken from an f32 surrogate of the same graph
    # while the reported loss stays in native precision.
    surrogate = width if width < 32 else None
    if surrogate is not None:
        LOGGER.info(f"Tuning with f32 gradient surrogate for f{width}")
    grad_dtype = torch.float32 if surrogate is not None else dtype

    domain_t = torch.linspace(*domain, samples, dtype=dtype)
    model = TunableIRModule(approximation.variables, approximation.expr)
    criterion = nn.MSELoss()
    baseline = tensor(
        evaluate(
            base_exp.expr, {v.result: domain_t.numpy() for v in base_exp.variables}
        )
    ).to(grad_dtype)
    domain_g = domain_t.to(grad_dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def native_loss() -> float:
        with torch.no_grad():
            out = model._eval((domain_t,), dtype)
            return float(criterion(out.to(grad_dtype), baseline))

    epochs = 600
    t0 = time.time()
    initial_loss = native_loss()
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
        out = model._eval((domain_g,), grad_dtype, surrogate)
        loss_g = criterion(out, baseline)
        loss_g.backward()
        optimizer.step()
        # native-precision loss; for f32 the gradient forward is the native
        # forward, so reuse it rather than running a second forward
        loss = loss_g.detach().item() if surrogate is None else native_loss()
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


def surrogate_bc(width: int) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """f16 bitcast semantics on f32 tensors, with f32-exact gradients.

    Returns (i2f, f2i): `i2f` takes an f32 tensor holding the integer bit
    pattern and returns the corresponding f16 value as f32; `f2i` takes an
    f32 tensor holding an f16-range value and returns its f16 bit pattern as
    f32. The gradients use the same analytic ramps as `make_bc` but computed
    in f32, where they do not underflow.
    """
    m_bits, B = _IEEE[width]
    L = 2**m_bits
    # tensor types wide enough to hold the pattern exactly
    it = torch.int32 if width <= 32 else torch.int64
    ft = {16: torch.float16, 32: torch.float32, 64: torch.float64}[width]

    class I2F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> Any:
            ctx.save_for_backward(x)
            P = x.to(it)
            sign = (P >> (width - 1)) & 1
            e = (P >> m_bits) & ((1 << (width - 1 - m_bits)) - 1)
            m = P & (L - 1)
            m = m.to(torch.float32)
            e = e.to(torch.float32)
            val = torch.where(
                e == 0,
                m * (2.0 ** (-(B + m_bits))),
                (L + m) * torch.pow(2.0, e - (B + m_bits)),
            )
            return torch.where(sign == 1, -val, val)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            x = ctx.saved_tensors[0]
            return grad_output * torch.pow(2.0, torch.floor(x / L) - (B + m_bits))

    class F2I(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> Any:
            ctx.save_for_backward(x)
            v = x.to(ft)
            P = v.view(getattr(torch, f"int{width}")).to(it) & ((1 << width) - 1)
            return P.to(torch.float32)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            x = ctx.saved_tensors[0]
            # within an exponent bin dP/dv = 2^(m_bits + B - e)
            e = torch.floor(torch.log2(x.abs().clamp(min=2.0 ** -62))) + B
            return grad_output * torch.pow(2.0, m_bits + B - e)

    return I2F.apply, F2I.apply
