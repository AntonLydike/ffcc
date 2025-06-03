import itertools
from collections.abc import Sequence, Iterator
import time
import numpy as np
from aalib.duration import duration
from aalib.progress import simple_progress
from ffcc.ir import TunableNode, IntType
from ffcc.helper import step_float
from ffcc.jit import Program


class GridSearch(Iterator[tuple[int | float, ...]]):
    tunes: Sequence[TunableNode]
    steps: int
    float_step_size: int

    # internal vars
    _d: int
    _idx: list[int]

    def __init__(
        self,
        tunes: Sequence[TunableNode],
        steps_per_dim: int,
        float_step_size: int = 512,
    ):
        self.tunes = tunes
        self.steps = steps_per_dim
        self.float_step_size = float_step_size

        self._d = len(tunes)
        self._idx = [0] * len(tunes)

    def __len__(self) -> int:
        return self.steps ** len(self.tunes)

    def conf_at_point(self, pt: Sequence[int]) -> tuple[int | float, ...]:
        return tuple(self._conf_for_dim(i, pos) for i, pos in enumerate(pt))

    def _conf_for_dim(self, dim: int, pos: int) -> int | float:
        t = self.tunes[dim]
        if isinstance(t.type, IntType):
            return t.hint + (pos - (self.steps // 2))
        else:
            return step_float(t.hint, (pos - (self.steps // 2)) * self.float_step_size)

    def __next__(self):
        if self._idx[-1] >= self.steps:
            raise StopIteration()

        c = self.conf_at_point(self._idx)

        i = 0
        self._idx[i] += 1
        while self._idx[i] >= self.steps and i < len(self.tunes) - 1:
            self._idx[i] = 0
            self._idx[i + 1] += 1
            i += 1
        return c


class GreedyDescent:
    def __init__(
        self,
        p: Program,
        domain: np.ndarray,
        ref: np.ndarray,
        int_step_size: int = 32,
        float_step_size=512,
        min_float_step_size=10,
        max_steps: int = 100_000,
        epsilon: float = 0.0,
    ):
        self.program = p
        self.domain = domain
        self.ref = ref

        self.current_conf = list(p.initial_tune)
        self.epsilon = epsilon
        self.current_err = p.max_relative_error(ref, domain, epsilon, self.current_conf)

        self._int_step_size = int_step_size
        self._float_step = float_step_size
        self._min_float_step = min_float_step_size
        self.max_steps = max_steps
        self.seen = set()

        self.step = 0

    def take_step(self) -> bool:
        self.step += 1

        best_cfg = self.current_conf
        best_err = self.current_err
        best_dir_vec = [0] * len(best_cfg)
        cfgs = set()

        for dir_vec in itertools.product((-1, 0, 1), repeat=len(self.current_conf)):
            tune = self._step_cfg(dir_vec)
            cfgs.add(tune)
            if tune in self.seen:
                continue
            err = self.program.max_relative_error(
                self.ref, self.domain, self.epsilon, tune
            )
            if err < best_err:
                best_err = err
                best_cfg = tune
                best_dir_vec = dir_vec

        self.seen = cfgs

        # if no config change happened
        if best_cfg == self.current_conf:
            # pick a dim to reduce step size in
            for i, d, cfg in zip(range(len(best_cfg)), best_dir_vec, best_cfg):
                if d == 0:
                    continue
                elif isinstance(cfg, int) and self._int_step_size > 1:
                    self._int_step_size = self._int_step_size // 2
                    return True
                elif self._float_step > self._min_float_step:
                    # half float step range
                    self._float_step = self._float_step // 2
                    return True
            return False

        self.current_conf = best_cfg
        self.current_err = best_err
        return True

    def run(self, progress: bool = True) -> list[int | float]:
        t0 = time.time()
        while self.step < self.max_steps:
            if not self.take_step():
                return self.current_conf
            if progress:
                simple_progress(
                    self.step, self.max_steps, t0, f"loss={self.current_err:.3g}"
                )
        if progress:
            print()
        dur = time.time() - t0
        print(f"took {duration(dur)}")
        return self.current_conf

    def _step_cfg(self, direction: tuple[int, ...]) -> tuple[int | float, ...]:
        result = []
        for d, tune, current_val in zip(
            direction, self.program.tunables, self.current_conf
        ):
            if d == 0:
                result.append(current_val)
            elif isinstance(tune.type, IntType):
                result.append(current_val + int(d * self._int_step_size))
            else:
                result.append(step_float(current_val, int(d * self._float_step)))
        return tuple(result)
