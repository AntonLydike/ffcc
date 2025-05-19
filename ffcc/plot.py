import argparse
from email.policy import default
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import sys

from ffcc.jit import Program
from ffcc.opt.simplify import simp
from ffcc.opt.approximate import approx
from ffcc.cse import cse
from ffcc.main import open_source, config_log
import time

from ffcc.parse import parse_ssa

matplotlib.use("TkAgg")


# TODO: unused -- implement this
def plot_error_surface(
    program: Program,
    zlim=1,
    resolution=500,
    err_samples=10_000_000,
):

    domain = np.linspace(1, 4, err_samples)
    ref = domain ** (-0.5)
    tunables = p.initial_tune

    print("Plotting data...")
    print("Evaluating {} spots".format(err_samples * resolution ** len(tunables)))
    start_time = time.time()
    error_grid = p.sweep_tunables(
        ref,
        domain,
        [(t * 0.9, t * 1.1) for t in tunables],
        [resolution] * len(tunables),
    )
    print("Took {}s".format(time.time() - start_time))

    # clamp to zlim
    error_grid[error_grid > zlim] = np.nan

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    x, y = (np.linspace(t - 0.4, t + 0.4, resolution) for t in tunables)

    surf = ax.plot_surface(x, y, error_grid, cmap="viridis", zorder=10)

    ## Add slice along sigma axis (fixed threehalfs)
    # err_sigma_slice = [limit(max_relative_error(s, threehalfs_ref, num_points=err_sampling)) for s in sigma_vals]
    # ax.plot(sigma_vals, [threehalfs_ref] * resolution[0], err_sigma_slice, color='darkblue', linewidth=2, alpha=0.8, label=f'$\\text{{threehalfs}}={threehalfs_ref}$', zorder=20)

    ## Add slice along threehalfs axis (fixed sigma)
    # err_threehalfs_slice = [limit(max_relative_error(sigma_ref, t, num_points=err_sampling)) for t in threehalfs_vals]
    # ax.plot([sigma_ref] * resolution[1], threehalfs_vals, err_threehalfs_slice, color='orange', linewidth=2, alpha=0.8, label=f'$\\sigma={sigma_ref}$', zorder=20)

    ax.set_xlabel(p.tunables[0].name)
    ax.set_ylabel(p.tunables[1].name)
    ax.set_zlabel("Max Relative Error")
    ax.set_zlim([np.min(error_grid), zlim])
    # ax.legend()

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    # plt.savefig("error_surface.pdf", format='pdf')

    plt.show()


def plot_eval(
    programs: Sequence[Program],
    x_domain: tuple[float, float],
    steps: int = 10000,
    logx: bool = False,
    logy: bool = False,
    names: tuple[str, ...] = None,
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
        p.eval_on_domain(domain, result=result)

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

    if len(programs) > 1:
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
    )

    if ns.gui:
        plt.show()

    if ns.output is not None:
        plt.savefig(ns.output)

    if ns.output is None and not ns.gui:
        print("Warning: No output format configured.")


if __name__ == "__main__":
    plot_main()
