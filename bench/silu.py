from ffcc.parse import parse_ssa
from ffcc.jit import Program
from aalib.duration import duration

import numpy as np

for num_elems in (100, 100_000, 100_000_000):

    x = np.linspace(-4, 4, num_elems, dtype=np.float32)
    # compute reference silu:
    expected =  x / (1 + np.exp(-x))

    orig = Program(parse_ssa("""
    %x = var 'x' : f32
    %one = constant 1 : f32
    %e = constant 2.718281828459045 : f32
    %mx = negate %x : f32
    %exp = pow %e, %mx : f32
    %one_p_ex = add %one, %exp : f32
    %d = div %one, %one_p_ex : f32
    %res = mul %x, %d : f32
    """), num_threads=1)

    p = Program(node := parse_ssa("""%x = var 'x' : f32
    %sigma = tunable 's2' = 2129881071 : i32
    %one = tunable 'one' = 1.0119495391845703 : f32
    %sigma1 = tunable 's1' = 1065039594 : i32
    %0 = tunable 'L' = 12000943.0 : f32
    %mx = negate %x : f32
    %1 = mul %0, %mx : f32
    %2 = add %sigma1, %1 : i32
    %3 = bitcast i2f %2 to f32
    %one_p_ex = add %one, %3 : f32
    %4 = bitcast f2i %one_p_ex to i32
    %5 = sub %sigma, %4 : i32
    %y = bitcast i2f %5 to f32
    
    // x * y * y
    %_0 = mul %one_p_ex, %y : f32
    %_1 = mul %_0, %y : f32
    
    // 2 * y
    %c2 = tunable two = 2 : f32
    %_2 = mul %c2, %y : f32
    %yn = sub %_2, %_1 : f32
    %r = mul %x, %yn : f32
    """), num_threads=1)


    scalart = Program(node, num_threads=4)

    pavx = Program(node, vectorise=8, num_threads=1)
    pavxt = Program(node, vectorise=8, num_threads=4)

    def bench_impl(impl: Program, domain: np.ndarray):
        result = np.zeros_like(domain)
        #print("benchmarking impl:")
        import timeit
        iters, time = timeit.Timer(lambda : impl.eval_on_domain(domain, result=result)).autorange()
        print("mem: {}/iter, {}/elem".format(duration(time/iters), duration(time / (iters * domain.size))))

    def bench_impl_sp(impl: Program, domain: np.ndarray):
        result = np.zeros_like(domain)
        #print("benchmarking impl:")
        import timeit
        iters, time = timeit.Timer(lambda : impl.eval_on_linspace(domain[0], domain[-1], domain.size, result=result)).autorange()
        print("cmp: {}/iter, {}/elem".format(duration(time/iters), duration(time / (iters * domain.size))))

    def bench_max_rel_err(impl: Program, domain: np.ndarray, expected: np.ndarray):
        print("benchmarking max_rel_err:")
        import timeit
        iters, time = timeit.Timer(lambda : impl.max_relative_error(expected, domain)).autorange()
        print("{}/iter, {}/elem".format(duration(time/iters), duration(time / (iters * domain.size))))
    import timeit

    print(f"-----\nnum_elms = {num_elems}")
    print("numpy:")
    iters, time = timeit.Timer(lambda : 1/(1+np.exp(-x))).autorange()
    print("{}/iter, {}/elem".format(duration(time/iters), duration(time / (iters * num_elems))))
    print('exact compiled:')
    bench_impl(orig, x)
    bench_impl_sp(orig, x)
    print('scalar:')
    bench_impl(p, x)
    bench_impl_sp(p, x)
    print("scalar+4threads:")
    bench_impl(scalart, x)
    bench_impl_sp(scalart, x)
    print('avx:')
    bench_impl(pavx, x)
    bench_impl_sp(pavx, x)
    print('avx+4threads:')
    bench_impl(pavxt, x)
    bench_impl_sp(pavxt, x)
