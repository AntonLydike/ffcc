import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import sys

from ffcc.jit import Program
from ffcc.main import open_source, config_log

from ffcc.parse import parse_ssa

matplotlib.use("TkAgg")


def plot_eval(
    programs: Sequence[Program],
    x_domain: tuple[float, float],
    steps: int = 10000,
    logx: bool = False,
    logy: bool = False,
    names: tuple[str, ...] = None,
    hide_legend: bool = False,
    tune: tuple[int | float, ...] = None,
) -> plt.Axes:

    if logx:
        domain = np.logspace(
            math.log10(x_domain[0]),
            math.log10(x_domain[1]),
            steps,
            base=10,
            dtype=np.float32,
        )
    else:
        domain = np.linspace(x_domain[0], x_domain[1], steps, dtype=np.float32)

    result = np.zeros_like(domain)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    if names is None:
        names = [f"#{i}" for i in range(len(programs))]

    linestyles = ("-", "--", ":")

    for p, name, i in zip(programs, names, range(len(names)), strict=True):
        p.eval_on_domain(domain, tunables=tune, result=result)

        ax.plot(
            domain,
            result,
            label=name,
            ls=linestyles[i % len(linestyles)],
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if len(programs) > 1 and not hide_legend:
        ax.legend()

    return ax


def parse_domain(dom: str) -> tuple[float, float]:
    parts = dom.split(",")
    if len(parts) != 2:
        raise ValueError("Expected a single comma in the domain")
    return float(parts[0]), float(parts[1])


def plot_main():
    parser = argparse.ArgumentParser(description="Plot program")
    parser.add_argument(
        "-i",
        "--input",
        help="Input to read from (default stdin)",
        type=open_source(sys.stdin),
        default=sys.stdin,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to write to (pdf, png supported)",
        default=None,
    )
    parser.add_argument(
        "-s", "--steps", type=int, help="Number of steps on the x-axsis", default=10000
    )
    parser.add_argument("-v", "--verbose", help="Enable loggin", action="store_true")
    parser.add_argument(
        "--gui", help="Open result in interactive window", action="store_true"
    )
    parser.add_argument(
        "--logx", help="Plot x-axis as log-scale", action="store_true", default=False
    )
    parser.add_argument(
        "--logy", help="Plot y-axis as log-scale", action="store_true", default=False
    )
    parser.add_argument(
        "--names",
        help="Labels of plots, comma separated",
        default=None,
        type=lambda s: [x.strip() for x in s.split(",")],
    )
    parser.add_argument(
        "domain",
        help="Domain to evaluate on as two comma separated floats",
        type=parse_domain,
    )
    parser.add_argument(
        "--hide-legend",
        help="Hide the legend",
        action="store_true",
    )

    ns = parser.parse_args()

    config_log(ns.verbose, False)

    if ns.input == sys.stdin and sys.stdin.isatty():
        sys.stderr.write(">> Waiting for input...\n")

    parts = []
    lineno = 1
    for part in ns.input.read().split("-----"):
        parts.append((part, lineno))
        lineno += part.count("\n")

    programs = [Program(parse_ssa(part, lineno)) for part, lineno in parts]

    plot_eval(
        programs,
        x_domain=ns.domain,
        steps=ns.steps,
        logx=ns.logx,
        logy=ns.logy,
        names=ns.names,
        hide_legend=ns.hide_legend,
    )

    if ns.gui:
        plt.show()

    if ns.output is not None:
        plt.savefig(ns.output)

    if ns.output is None and not ns.gui:
        print("Warning: No output format configured.")


if __name__ == "__main__":
    plot_main()
