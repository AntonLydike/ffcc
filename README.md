# Fast Math Program Synthesis Tool:

Tool for generating fast approximations of floating point math by (ab)using the floating point representation.

This tool was produced for the ICML paper "Faster Activation Functions at the Edge for Post-Training Speedups" [link](https://openreview.net/forum?id=u2o6hPE8ik).

## Usage:

Basic usage (`ffcc` tool):

```bash
$ ffcc -e "silu(x) = x / (1 + exp(-x))" --approx=exp --tune=[-6,6] -o torch
```

This should give you, after some tuning time, the following torch module:
```python
import torch
from torch import nn, Tensor


class FastSilu(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
                v0 = (-12104087.321284886 * x)
                v1 = (1064872480.0625 + v0)
                v2 = v1.type(torch.int32).view(torch.float32)
                return (x / (1.0 + v2))
```

Flags explained:
- `-e $expr` provides the input expression to approximate
- `-approx=exp` approximates exponentiation (`log` and `div` can be added as well, though `div` support is experimental)
- `--type f16` targets half precision (default is `f32`); the bitcast trick works for any IEEE float width
- `-tune=[-6,6]` performs gradient-descent based constant tuning on the domain $[-6,6]$
- `-o torch` prints the resulting code as a pytorch module

Tunable constants are re-parameterized as O(1) values, so the same learning
rate works across float widths. For sub-f32 targets (e.g. f16), gradients
are computed through an f32 surrogate of the bitcast graph: the f16 forward
is exact, and the f32 surrogate's analytic backward avoids the underflow
that zeroing f16 gradients (and thus breaking Adam) would cause.

## Development Environment:

There's a `flake.nix` file for all nixos users.

The python dependencies are managed through `uv`. Setting everything up usually involves
running a combination of `uv venv; uv sync --all-extras; source .venv/bin/activate`.

To run tests, use `lit tests/filecheck`, there are no pytests yet.

There is an `ffcc-opt` tool available for testing.

## License:

ffcc - fast float compiler - Copyright (C) 2025 Anton Lydike

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
