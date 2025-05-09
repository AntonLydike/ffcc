from __future__ import annotations

import sys
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import TextIO, Callable

from synth.ir import IRNode

from synth.cse import cse
from synth.parse import parse_ssa
from synth.printer import print_dag, print_ssa
from synth.rewrite.simplify import simp
from synth.rewrite.approximate import approx

passes = {
    'cse': cse,
    'simp': simp,
    'approx': approx,
}

def open_source(dash:TextIO) -> Callable[[str], TextIO]:
    def parse(src: str):
        if src == '-':
            return dash
        return open(src, 'r')
    return parse

def get_passes(pass_args: str) -> list[Callable[[str], IRNode | None]]:
    pipeline = []
    for arg in pass_args.split(','):
        if arg not in passes:
            raise ValueError("Unknown pass", arg)
        pipeline.append(passes[arg])
    return pipeline

formatter = {
    'dag': print_dag,
    'ssa': print_ssa,
}


@dataclass
class Main:
    input: TextIO
    out: TextIO

    split: bool

    out_formatter: Callable[[IRNode, TextIO], None]

    passes: list[Callable[[IRNode], IRNode]]

    split_on: str = '-----'

    verbose: bool = False
    log_to_out: bool = False

    @classmethod
    def from_cli(cls, cli: list[str]) -> Main:
        parser = ArgumentParser(prog="synth")
        parser.add_argument('input', help="source file, - for stdin", default=sys.stdin, type=open_source(sys.stdin), nargs='?')
        parser.add_argument("-o", '--output', help="dest file, - for stdout", default=sys.stdout, type=open_source(sys.stdout))
        parser.add_argument('-p', '--passes', help="passes to apply", default=[], type=get_passes)
        parser.add_argument('-f', '--format', help="output format", default=print_ssa, choices=formatter, type=lambda x: formatter[x])
        parser.add_argument('--split-input-file', help="split input files on -----", action='store_true', default=False)
        parser.add_argument('--split-on', type=str, help="boundary to split on (default -----)", default='-----')
        parser.add_argument('--verbose', action='store_true', default=False, help="Print verbose output")
        parser.add_argument('--log-to-out', action='store_true', default=False, help="Log to output stream")

        ns = parser.parse_args(args=cli[1:])
        return Main(input=ns.input, out=ns.output, passes=ns.passes, out_formatter=ns.format, split=ns.split_input_file, split_on=ns.split_on, verbose=ns.verbose, log_to_out=ns.log_to_out)


    def apply(self):
        log_conf = {}
        if self.verbose:
            log_conf['level']=logging.INFO
        if self.log_to_out:
            log_conf['stream'] = self.out
        if log_conf:
            logging.basicConfig(**log_conf)


        if self.input == sys.stdin and sys.stdin.isatty():
            sys.stderr.write(">> Waiting for input...\n")
        if self.split:
            parts = []
            lineno = 1
            for part in self.input.read().split(self.split_on):
                parts.append((part, lineno))
                lineno += part.count('\n')
        else:
            parts = [(self.input.read(), 1)]

        for i, (part, lineno) in enumerate(parts):
            ir = parse_ssa(part, lineno)
            for p in self.passes:
                ir = p(ir)
            self.out_formatter(ir, self.out)
            if i < len(parts) - 1:
                self.out.write(f'\n// {self.split_on}\n\n')

    def __call__(self):
        self.apply()
