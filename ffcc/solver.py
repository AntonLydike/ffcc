from z3 import *
import numpy as np


# --------------------------------------------------------------------
# Helper: bitcast helpers work on both old (<4.12) and new Z3 releases
# --------------------------------------------------------------------
def _bitcast_i2f(bv_expr, fp_sort):
    """Bit-cast 32-bit BitVec → float32 (keeping the bits)."""
    try:
        return fpFromIEEEBV(bv_expr, fp_sort)  # Z3 ≥ 4.12
    except NameError:
        return fpBVToFP(bv_expr, fp_sort)  # Older Z3


def _bitcast_f2i(fp_expr):
    """Bit-cast float32 → 32-bit BitVec (keeping the bits)."""
    return fpToIEEEBV(fp_expr)  # Available in all versions


class SMTSolver:
    """
    Build an SMT problem that contains both floating-point and integer
    operations and asks Z3 to find values for tunable constants so
    that the final floating-point result is within |delta|.
    """

    # Common IEEE-754 and BitVec sorts
    FP32 = FPSort(8, 24)  # float32
    BV32 = BitVecSort(32)  # 32-bit bit-vector (signed/unsigned i32)
    RND = RNE()  # Default rounding mode: round-nearest-even

    def __init__(self):
        self.solver = Solver()
        self.env = {}  # name → Z3 expression

    # ----------------------------------------------------------------
    #  Variable declarations
    # ----------------------------------------------------------------
    def add_tunable(self, name, sort):
        if sort == "float":
            v = FP(name, self.FP32)
        elif sort == "int":
            v = BitVec(name, 32)
        else:
            raise ValueError("sort must be 'float' or 'int'")
        self.env[name] = v
        self.forbid_nan(name)

    def add_const(self, name, val, sort):
        if sort == "float":
            v = FP(name, self.FP32)
        elif sort == "int":
            v = BitVec(name, 32)
        self.solver.add(v == val)
        self.env[name] = v

    def add_inp(self, name, value, sort="float"):
        """Add a non-tunable (fixed) input."""
        if sort == "float":
            self.env[name] = FPVal(value, self.FP32)
        else:
            self.env[name] = BitVecVal(value, 32)

    # ----------------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------------
    def _expr(self, name):
        if name not in self.env:
            assert not isinstance(name, str)
            return name
        return self.env[name]

    def _kind(self, name):
        if not isinstance(name, str):
            return "float" if isinstance(name, float) else "int"
        return (
            "float" if self._expr(name).sort_kind() == Z3_FLOATING_POINT_SORT else "int"
        )

    def forbid_nan(self, name):
        v = self.env[name]
        if self._kind(name) != "float":
            exp = Extract(30, 23, v)  # 8-bit exponent
            self.solver.add(exp != BitVecVal(0xFF, 8))  # 0xFF would mean NaN or Inf
        else:
            self.solver.add(Not(fpIsNaN(v)))

    # ----------------------------------------------------------------
    #  Operations (mirroring the IR instructions)
    # ----------------------------------------------------------------
    def add_neg(self, dst, src):
        if self._kind(src) != "float":
            raise TypeError("negate only supports float")
        self.env[dst] = fpNeg(self._expr(src))

    def add_mul(self, dst, lhs, rhs):
        if self._kind(lhs) == "float":
            self.env[dst] = fpMul(self.RND, self._expr(lhs), self._expr(rhs))
        else:
            self.env[dst] = self._expr(lhs) * self._expr(rhs)

    def add_add(self, dst, lhs, rhs):
        if self._kind(lhs) == "float":
            self.env[dst] = fpAdd(self.RND, self._expr(lhs), self._expr(rhs))
        else:
            self.env[dst] = self._expr(lhs) + self._expr(rhs)

    def add_sub(self, dst, lhs, rhs):
        if self._kind(lhs) == "float":
            self.env[dst] = fpSub(self.RND, self._expr(lhs), self._expr(rhs))
        else:
            self.env[dst] = self._expr(lhs) - self._expr(rhs)

    def add_ashr(self, dst, lhs, shift):
        if self._kind(lhs) != "int":
            raise TypeError("ashr is only valid for integers")
        self.env[dst] = self._expr(lhs) >> self._expr(shift)

    # numeric FP → signed int32
    def add_f2s(self, dst, src):
        if self._kind(src) != "float":
            raise TypeError("fptosi expects a float source")
        self.env[dst] = fpToSBV(self.RND, self._expr(src), self.BV32)

    # bit-casts (keep the bit pattern)
    def add_bitcast_f2i(self, dst, src):
        if self._kind(src) != "float":
            raise TypeError("bitcast f2i expects a float source")
        self.env[dst] = _bitcast_f2i(self._expr(src))

    def add_bitcast_i2f(self, dst, src):
        if self._kind(src) != "int":
            raise TypeError("bitcast i2f expects an int source")
        self.env[dst] = _bitcast_i2f(self._expr(src), self.FP32)

    # ----------------------------------------------------------------
    #  Output constraint
    # ----------------------------------------------------------------
    def add_output_constraint(self, expr_name, target_val, delta):
        if self._kind(expr_name) != "float":
            raise TypeError("output expression must be float")
        expr = self._expr(expr_name)
        target = FPVal(target_val, self.FP32)
        error = fpAbs(fpSub(self.RND, expr, target))
        self.solver.add(fpLEQ(error, FPVal(delta, self.FP32)))

    # ----------------------------------------------------------------
    #  Solve
    # ----------------------------------------------------------------
    def solve(self):
        if self.solver.check() != sat:
            return None
        model = self.solver.model()
        return {n: model.eval(e, model_completion=True) for n, e in self.env.items()}


class SMTSynthesizer:
    def __init__(self, prog, low, high, sample_num=10000):
        self.prog = prog
        self.low = low
        self.high = high
        self.keys = self.gen_keys()

        self.initial_delta = 0.05
        self.threshold = 1e-4
        self.max_steps = 1000
        self.epsilon = 1e-8
        self.sample_num = sample_num

    # TODO: Automate this function
    def gen_keys(self):
        if self.prog.name == "qsqrt":
            return [("sigma", "int")]
        elif self.prog.name == "sigmoid":
            return [("sigma", "int"), ("sigma1", "int")]
        assert 0

    def criterion(self, gt, pred):
        return np.abs(gt - pred) / (gt + self.epsilon)

    # TODO: Automate this function
    def find_conflict(self, tunable, delta):
        checkpoints = np.random.uniform(
            low=self.low, high=self.high, size=self.sample_num
        ).astype(np.float32)
        if self.prog.name == "qsqrt":
            refs = 1 / np.sqrt(checkpoints)
        elif self.prog.name == "sigmoid":
            refs = 1 / (1 + np.exp(-checkpoints))
        else:
            assert 0
        result = np.zeros_like(checkpoints).astype(np.float32)
        self.prog.eval_on_domain(checkpoints, tunables=tunable, result=result)
        loss = self.criterion(refs, result)
        max_loss, max_idx = np.max(loss), np.argmax(loss)
        if max_loss >= delta:
            return [checkpoints[max_idx].item(), refs[max_idx].item()], max_loss
        return None, max_loss

    # TODO: Automate this function
    def gen_initial_io_pairs(self):
        checkpoints = np.random.uniform(low=self.low, high=self.high, size=1)
        if self.prog.name == "qsqrt":
            refs = 1 / np.sqrt(checkpoints)
        elif self.prog.name == "sigmoid":
            refs = 1 / (1 + np.exp(-checkpoints))
        else:
            assert 0
        return [[checkpoints.item(), refs.item()]]

    # TODO: Automate this function
    def parse(self, solver, idx):
        if self.prog.name == "qsqrt":
            solver.add_bitcast_f2i("t1_%d" % idx, "x_%d" % idx)
            solver.add_ashr("t2_%d" % idx, "t1_%d" % idx, 1)
            solver.add_sub("t3_%d" % idx, 0, "t2_%d" % idx)
            solver.add_add("t4_%d" % idx, "sigma", "t3_%d" % idx)
            solver.add_bitcast_i2f("r_%d" % idx, "t4_%d" % idx)
        elif self.prog.name == "sigmoid":
            solver.add_const("one_%d" % idx, 1.0, "float")
            solver.add_const("t0_%d" % idx, 12102203.161561485, "float")
            # %mx = negate %x : f32
            solver.add_neg("mx_%d" % idx, "x_%d" % idx)
            # %1 = mul %0, %mx : f32
            solver.add_mul("t1_%d" % idx, "t0_%d" % idx, "mx_%d" % idx)
            # Need cvt float to int ?
            solver.add_f2s("t11_%d" % idx, "t1_%d" % idx)
            # %2 = add %sigma1, %1 : i32
            solver.add_add("t2_%d" % idx, "sigma1", "t11_%d" % idx)
            # %3 = bitcast i2f %2 to f32
            solver.add_bitcast_i2f("t3_%d" % idx, "t2_%d" % idx)
            # %one_p_ex = add %one, %3 : f32
            solver.add_add("one_p_ex_%d" % idx, "one_%d" % idx, "t3_%d" % idx)
            # %4 = bitcast f2i %one_p_ex to i32
            solver.add_bitcast_f2i("t4_%d" % idx, "one_p_ex_%d" % idx)
            # %5 = sub %sigma, %4 : i32
            solver.add_sub("t5_%d" % idx, "sigma", "t4_%d" % idx)
            # %r = bitcast i2f %5 to f32
            solver.add_bitcast_i2f("r_%d" % idx, "t5_%d" % idx)

    def sat(self, io_pairs, delta):
        solver = SMTSolver()
        for name, type_ in self.keys:
            solver.add_tunable(name, type_)
        for idx, (inp, out) in enumerate(io_pairs):
            solver.add_inp("x_%d" % idx, inp, "float")
            self.parse(solver, idx)
            solver.add_output_constraint(
                "r_%d" % idx, out, delta=delta * (out + self.epsilon)
            )
        res = solver.solve()
        if res == None:
            return None
        tunable = []
        for name, type_ in self.keys:
            assert type_ == "int"
            tunable.append(res[name].as_long())
        return tunable

    def synthesize(self):
        delta = self.initial_delta
        io_pairs = self.gen_initial_io_pairs()
        step = 0
        best_delta = 1
        lower_bound = 0
        best_tunable = None

        while True:
            tunable = self.sat(io_pairs, delta)
            if not tunable:
                new_delta = (best_delta + delta) / 2
                lower_bound = delta
            else:
                new_io_pair, loss = self.find_conflict(tunable, delta)
                if new_io_pair:
                    io_pairs.append(new_io_pair)
                    new_delta = delta
                else:
                    best_delta = delta
                    best_tunable = tunable
                    new_delta = (delta + lower_bound) / 2
            approx_error_bound = best_delta - lower_bound
            print("---- Step %d ----" % step)
            print("io_pairs      : ", len(io_pairs))
            print("delta         :", delta)
            print("best delta    :", best_delta)
            print("lower bound   :", lower_bound)
            print("error bound   :", approx_error_bound)
            print("Best Tunable  :", best_tunable)
            if np.abs(approx_error_bound) < self.threshold:
                break
            delta = new_delta
            step += 1
            if step == self.max_steps:
                break
        return best_tunable


from ffcc.parse import parse_ssa
from ffcc.jit import Program


def test_qsqrt(low=0.01, high=2, sample_num=10000):
    prog = Program(
        node := parse_ssa(
            """
        %sigma = tunable 'sigma' = 1597463007 : i32
        %x = var 'x' : f32
        %0 = bitcast f2i %x to i32
        %1 = constant 1 : i32
        %2 = ashr %0, %1 : i32
        %3 = negate %2 : i32
        %4 = add %sigma, %3 : i32
        %5 = bitcast i2f %4 to f32
        """
        )
    )
    prog.name = "qsqrt"
    syn = SMTSynthesizer(prog, low, high, sample_num)
    syn_tunable = syn.synthesize()

    # Evaluate
    base_tunable = [1597463007]
    x = np.linspace(low, high, sample_num, dtype=np.float32)
    expected = 1 / np.sqrt(x)

    syn_result = np.zeros_like(x).astype(np.float32)
    prog.eval_on_domain(x, tunables=syn_tunable, result=syn_result)
    syn_loss = np.max(syn.criterion(expected, syn_result))

    base_result = np.zeros_like(x).astype(np.float32)
    prog.eval_on_domain(x, tunables=base_tunable, result=base_result)
    base_loss = np.max(syn.criterion(expected, base_result))

    print("Baseline loss :", base_loss)
    print("Synth loss    :", syn_loss)


def test_sigmoid(low=-5, high=5, sample_num=10000):
    prog = Program(
        node := parse_ssa(
            """
        %sigma = tunable 'sigma' = 2129950675 : i32
        %one = constant 1 : f32
        %sigma1 = tunable 'sigma' = 1064975338 : i32
        %0 = constant 12102203.161561485 : f32
        %x = var 'x' : f32
        %mx = negate %x : f32
        %1 = mul %0, %mx : f32
        %2 = add %sigma1, %1 : i32
        %3 = bitcast i2f %2 to f32
        %one_p_ex = add %one, %3 : f32
        %4 = bitcast f2i %one_p_ex to i32
        %5 = sub %sigma, %4 : i32
        %6 = bitcast i2f %5 to f32
        """
        )
    )
    prog.name = "sigmoid"
    syn = SMTSynthesizer(prog, low, high, sample_num)
    syn_tunable = syn.synthesize()

    # Evaluate
    base_tunable = [2129950675, 1064975338]
    x = np.linspace(low, high, sample_num, dtype=np.float32)
    expected = 1 / (1 + np.exp(-x))

    syn_result = np.zeros_like(x).astype(np.float32)
    prog.eval_on_domain(x, tunables=syn_tunable, result=syn_result)
    syn_loss = np.max(syn.criterion(expected, syn_result))

    base_result = np.zeros_like(x).astype(np.float32)
    prog.eval_on_domain(x, tunables=base_tunable, result=base_result)
    base_loss = np.max(syn.criterion(expected, base_result))

    print("Baseline loss :", base_loss)
    print("Synth loss    :", syn_loss)


if __name__ == "__main__":
    test_qsqrt()
    test_sigmoid()
