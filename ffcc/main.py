from __future__ import annotations

import ast
import sys
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import TextIO, Callable

from ffcc.diff import diff
from ffcc.ir import IRNode

from ffcc.cse import cse
from ffcc.parse import parse_ssa
from ffcc.print_llvm import print_llvm_func_for
from ffcc.print import print_dag, print_ssa
from ffcc.rewrite.instantiate import instantiate_pass
from ffcc.rewrite.optimize_types import types
from ffcc.rewrite.simplify import simp
from ffcc.rewrite.approximate import approx

passes = {
    "cse": cse,
    "simp": simp,
    "approx": approx,
    "types": types,
    "instantiate": instantiate_pass,
}


def open_source(dash: TextIO) -> Callable[[str], TextIO]:
    def parse(src: str):
        if src == "-":
            return dash
        return open(src, "r")

    return parse


def get_passes(pass_args: str) -> list[Callable[[str], IRNode | None]]:
    pipeline = []
    for arg in pass_args.split(","):
        if arg not in passes:
            raise ValueError("Unknown pass", arg)
        pipeline.append(passes[arg])
    return pipeline


formatter = {
    "dag": print_dag,
    "ssa": print_ssa,
    "llvm": lambda node, buf: print_llvm_func_for(node, "my_func", buf),
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
    input: TextIO
    out: TextIO

    split: bool

    out_formatter: Callable[[IRNode, TextIO], None]

    passes: list[Callable[[IRNode], IRNode]]

    split_on: str = "-----"

    verbose: bool = False
    log_to_out: bool = False

    @classmethod
    def from_cli(cls, cli: list[str]) -> Main:
        parser = ArgumentParser(prog="ffcc")
        parser.add_argument(
            "input",
            help="source file, - for stdin",
            default=sys.stdin,
            type=open_source(sys.stdin),
            nargs="?",
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

        ns = parser.parse_args(args=cli[1:])
        return Main(
            input=ns.input,
            out=ns.output,
            passes=ns.passes,
            out_formatter=formatter[ns.format],
            split=ns.split_input_file,
            split_on=ns.split_on,
            verbose=ns.verbose,
            log_to_out=ns.log_to_out,
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
            for p in self.passes:
                ir = p(ir)
            self.out_formatter(ir, self.out)
            if i < len(parts) - 1:
                self.out.write(f"\n// {self.split_on}\n\n")

    def __call__(self):
        self.apply()


def main():
    main = Main.from_cli(sys.argv)
    main()
