from __future__ import annotations

import re
import sys
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import TextIO, Callable, Any

from ffcc.opt.rewriter import Rewriter
from ffcc.print_torch import print_torch
from ffcc.ir import IRNode

from ffcc.cse import cse
from ffcc.parse import Expression, parse_expr, parse_ssa
from ffcc.print_llvm import print_llvm_func_for
from ffcc.print import print_dag, print_ssa
from ffcc.opt import types, simp, approx

passes: dict[str, Rewriter | Callable[[IRNode], IRNode | None]] = {
    "cse": cse,
    "simp": simp,
    "approx": approx,
    "types": types,
}


def open_source(dash: TextIO) -> Callable[[str], TextIO]:
    def parse(src: str):
        if src == "-":
            return dash
        return open(src, "r")

    return parse


def get_passes(pass_args: str) -> list[Callable[[str], IRNode | None]]:
    pipeline = []
    while (match := re.match(r"^([a-z]+)(\{[^}]+\})?,?", pass_args)) is not None:
        name = match.group(1)
        args = match.group(2)
        if name not in passes:
            raise ValueError("Unknown pass", name)
        p = passes[name]
        if isinstance(p, Rewriter):
            args = (args or "")[1:-1]
            p = p.with_args(p.parse_args(args))
        elif args is not None:
            raise ValueError(f"Pass {name} does not take argumetns")
        pipeline.append(p)
        pass_args = pass_args[len(match.group(0)) :]
    return pipeline


formatter = {
    "dag": print_dag,
    "ssa": print_ssa,
    "llvm": lambda node, buf, **kwargs: print_llvm_func_for(
        node,
        buf,
        kwargs.get("module_name", "my_func"),
        **kwargs,
    ),
    "torch": print_torch,
}


def config_log(verbose: bool, log_to_out: bool, out=sys.stderr):
    log_conf = {}
    if verbose:
        log_conf["level"] = logging.INFO
    if log_to_out:
        log_conf["stream"] = out
    if log_conf:
        logging.basicConfig(**log_conf)


@dataclass
class Main:
    input: TextIO | Expression
    out: TextIO

    split: bool

    out_formatter: Callable[[IRNode, TextIO, Any], Any]

    passes: list[Callable[[IRNode], IRNode]]

    split_on: str = "-----"

    verbose: bool = False
    log_to_out: bool = False

    print_between_passes: bool = False

    vectorize: int = 1

    @classmethod
    def from_cli(cls, cli: list[str]) -> Main:
        parser = ArgumentParser(prog="ffcc-opt")
        in_group = parser.add_mutually_exclusive_group()
        in_group.add_argument(
            "input",
            help="source file, - for stdin",
            default=sys.stdin,
            type=open_source(sys.stdin),
            nargs="?",
        )
        in_group.add_argument(
            "-e",
            "--expression",
            type=parse_expr,
            help="Input expression to use, in the form of 'name(args) = math'",
        )
        parser.add_argument(
            "-o",
            "--output",
            help="dest file, - for stdout",
            default=sys.stdout,
            type=open_source(sys.stdout),
        )
        parser.add_argument(
            "-p", "--passes", help="passes to apply", default=[], type=get_passes
        )
        parser.add_argument(
            "-f", "--format", help="output format", default="ssa", choices=formatter
        )
        parser.add_argument(
            "--print-between-passes",
            help="Print IR between passes",
            action="store_true",
        )
        parser.add_argument(
            "--split-input-file",
            help="split input files on -----",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--split-on",
            type=str,
            help="boundary to split on (default -----)",
            default="-----",
        )
        parser.add_argument(
            "--verbose", action="store_true", default=False, help="Print verbose output"
        )
        parser.add_argument(
            "--log-to-out",
            action="store_true",
            default=False,
            help="Log to output stream",
        )
        parser.add_argument(
            "--vectorize",
            default=1,
            type=int,
            help="Print vectorized LLVM IR with the specified number of elements. 1 (= default) turns vectorization off.",
        )
        parser.add_argument(
            "--name", default=None, help="Name of the output module (llvm, torch)"
        )

        ns = parser.parse_args(args=cli[1:])
        return Main(
            input=ns.input if "input" in ns else ns.expression,
            out=ns.output,
            passes=ns.passes,
            out_formatter=formatter[ns.format],
            split=ns.split_input_file,
            split_on=ns.split_on,
            verbose=ns.verbose,
            log_to_out=ns.log_to_out,
            print_between_passes=ns.print_between_passes,
            vectorize=ns.vectorize,
        )

    def apply(self):
        config_log(verbose=self.verbose, log_to_out=self.log_to_out, out=self.out)

        if self.input == sys.stdin and sys.stdin.isatty():
            sys.stderr.write(">> Waiting for input...\n")

        if self.split:
            parts = []
            lineno = 1
            for part in self.input.read().split(self.split_on):
                parts.append((part, lineno))
                lineno += part.count("\n")
        else:
            parts = [(self.input.read(), 1)]

        for i, (part, lineno) in enumerate(parts):
            ir = parse_ssa(part, lineno)

            if i > 0:
                self.out.write(f"\n// {self.split_on}\n\n")

            for p_i, p in enumerate(self.passes):
                if self.print_between_passes and p_i > 0:
                    self.out_formatter(ir, self.out)
                    self.out.write(f"\n// {self.split_on}\n\n")

                ir = p(ir)

            self.out_formatter(
                ir,
                self.out,
                vectorise=self.vectorize,
                expression=self.input if isinstance(self.input, Expression) else None,
            )

    def __call__(self):
        self.apply()


def main():
    main = Main.from_cli(sys.argv)
    main()
