from argparse import ArgumentParser
import ast
import sys

from ffcc.cse import cse
from ffcc.ir import IRNode, VarNode
from ffcc.opt.simplify import simp
from ffcc.opt_main import config_log, formatter, open_source, passes
from ffcc.parse import Expression, _parse_type, parse_expr, parse_ssa
from ffcc.opt import approximate


def parse_domain(dom: str) -> tuple[float, float]:
    """Parse a domain given as [a, b] into two floats"""
    if dom[0] != "[" or dom[-1] != "]" or dom.count(",") != 1:
        raise ValueError("Domain must be of format [a, b]", dom)
    lit = ast.literal_eval(dom)
    assert len(lit) == 2, "Domain literal must contain exactly two numbers"
    assert all(
        isinstance(v, (float, int)) for v in lit
    ), "Domain literal must contain exactly two numbers"
    return tuple(lit)


def main():
    parser = ArgumentParser("ffcc")
    inp_parser = parser.add_mutually_exclusive_group(required=True)
    inp_parser.add_argument("-e", "--expression", help="Input expression")
    inp_parser.add_argument(
        "-i",
        "--input",
        help="Input file containing IR, defautls to stdin",
        default=sys.stdin,
        type=open_source(sys.stdin),
    )
    parser.add_argument(
        "--type",
        default="f32",
        help="Expected type of the expression (only usable with --expression flag), e.g. f32 or f64 or i32",
    )
    parser.add_argument(
        "--approximate",
        type=approximate.Arguments.parse,
        help="Approximate, provide a comma-separated list of functions to approximate, available are exp, log and div",
    )
    parser.add_argument(
        "--tune",
        type=parse_domain,
        help='Tune the final approximation on a given domain (syntax is "[start,end]")',
    )
    parser.add_argument(
        "-o", "--output", help="Choose output format", choices=formatter, default="ssa"
    )
    parser.add_argument(
        "--expr-name", help="Name of the function or module that is returned"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print more information",
    )

    args = parser.parse_args()
    if args.verbose:
        config_log(args.verbose, False)

    if args.expression is not None:
        typ = _parse_type(0, args.type, 0, args.type)
        exp = parse_expr(args.expression, typ)
    else:
        ir = parse_ssa(args.input.read())
        if args.input is not sys.stdin:
            args.input.close()
        vars = tuple(set(v for v in ir.walk() if isinstance(v, VarNode)))
        exp = Expression("my_func", vars, ir)

    rewritten_ir = exp.expr.copy()
    if args.approximate:
        approx_pass = approximate.approx.with_args(args.approximate)
        simp_pass = simp.with_args(simp.args_t())
        rewritten_ir: IRNode = cse(simp_pass(approx_pass(cse(simp_pass(rewritten_ir)))))

    _vars = {var.name: var for var in rewritten_ir.walk() if isinstance(var, VarNode)}
    rewritten_expr = Expression(
        f"Fast{exp.name[0].upper()}{exp.name[1:]}",
        tuple(_vars[v.name] for v in exp.variables),
        rewritten_ir,
    )

    if args.tune:
        from ffcc.tune import tune

        tune(exp, rewritten_expr, args.tune)

    formatter[args.output](
        rewritten_ir,
        sys.stdout,
        sym_name=(rewritten_expr.name),
    )
