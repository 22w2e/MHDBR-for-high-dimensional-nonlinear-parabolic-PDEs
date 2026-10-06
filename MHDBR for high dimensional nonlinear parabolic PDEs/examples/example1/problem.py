#!/usr/bin/env python3
"""Example 1 MHDBR J=1,...,4 with fine-grid Euler startup values.

This file contains the Example-1 problem definition and command-line runner.
Use one of ex1MHDBRJ1Euler.py,...,ex1MHDBRJ4Euler.py as the direct entry point.
No plotting code is included.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mhdbr.euler_startup import Problem, TrainingResult, train_mhdbr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_problem(config: dict[str, Any]) -> Problem:
    dimension = int(config["d"])
    terminal_time = float(config["T"])
    sigma = float(config.get("sigma", dimension))
    x0 = torch.zeros(dimension, device=DEVICE)

    def sample_state(time_value: float, batch_size: int) -> torch.Tensor:
        if abs(time_value) < 1.0e-14:
            return torch.zeros(batch_size, dimension, device=DEVICE)
        return sigma * math.sqrt(time_value) * torch.randn(
            batch_size, dimension, device=DEVICE
        )

    def euler_transition(
        state: torch.Tensor, delta_w: torch.Tensor, delta: float
    ) -> torch.Tensor:
        del delta
        return state.unsqueeze(1) + sigma * delta_w

    def main_transition(
        state: torch.Tensor, cumulative_dw: torch.Tensor, elapsed: float
    ) -> torch.Tensor:
        del elapsed
        return state.unsqueeze(1) + sigma * cumulative_dw

    # These use only the prescribed terminal function g and its gradient.
    # No analytic solution at a nonterminal startup time is used.
    def terminal_y(state: torch.Tensor) -> torch.Tensor:
        exponent = torch.clamp(
            terminal_time + torch.sum(state, dim=-1, keepdim=True) / dimension,
            -30.0,
            30.0,
        )
        return torch.sigmoid(exponent)

    def terminal_z(state: torch.Tensor) -> torch.Tensor:
        y_value = terminal_y(state)
        component = sigma * y_value * (1.0 - y_value) / dimension
        return component.expand(-1, dimension)

    def generator(
        time_value: float,
        state: torch.Tensor,
        y_value: torch.Tensor,
        z_value: torch.Tensor,
    ) -> torch.Tensor:
        del time_value, state
        coefficient = y_value - (dimension + 2.0) / (2.0 * dimension)
        sum_z = torch.sum(z_value, dim=-1, keepdim=True)
        if config.get("use_original_generator_scaling", True):
            return coefficient * sum_z / sigma
        return coefficient * sum_z

    true_y0 = 0.5
    true_z0 = torch.full((dimension,), 0.25 * sigma / dimension, device=DEVICE)
    return Problem(
        name="Example 1",
        dimension=dimension,
        terminal_time=terminal_time,
        device=DEVICE,
        initial_state=x0,
        sample_state=sample_state,
        euler_transition=euler_transition,
        main_transition=main_transition,
        terminal_y=terminal_y,
        terminal_z=terminal_z,
        generator=generator,
        true_y0=true_y0,
        true_z0=true_z0,
    )


def parser_for_order(order: int) -> argparse.ArgumentParser:
    # Preserve the main-scheme defaults of the corresponding original files;
    # only the startup construction is changed here.
    n_defaults = {1: 20, 2: 40, 3: 30, 4: 30}
    batch_defaults = {1: 32, 2: 128, 3: 72, 4: 72}
    branch_defaults = {1: 100, 2: 400, 3: 225, 4: 225}
    seed_defaults = {1: 1812, 2: 1203, 3: 1023, 4: 1023}
    parser = argparse.ArgumentParser(
        description=(
            f"Example 1 MHDBR J={order} with nonterminal startup layers "
            "computed by refined Euler neural regression."
        )
    )
    parser.add_argument("--d", type=positive_int, default=200)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--N", type=positive_int, default=n_defaults[order])
    parser.add_argument(
        "--K-batch", type=positive_int, default=batch_defaults[order]
    )
    parser.add_argument(
        "--M-branch", type=positive_int, default=branch_defaults[order]
    )
    parser.add_argument("--hidden-layers", type=positive_int, default=2)
    parser.add_argument("--hidden-dim-offset", type=positive_int, default=110)
    parser.add_argument("--train-steps", type=positive_int, default=6000)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=seed_defaults[order])
    parser.add_argument(
        "--use-original-generator-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--warm-start-from-next",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lr-decay-fraction", type=float, default=0.6)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)

    parser.add_argument(
        "--startup-refinement",
        type=positive_int,
        default=None,
        help=(
            "Fine Euler substeps per coarse interval. The default computes "
            "the theoretical minimum R satisfying sqrt(h/R)<=h^J."
        ),
    )
    parser.add_argument("--startup-target-factor", type=float, default=1.0)
    parser.add_argument("--startup-train-steps", type=positive_int, default=6000)
    parser.add_argument("--startup-batch-size", type=positive_int, default=None)
    parser.add_argument("--startup-branches", type=positive_int, default=None)
    parser.add_argument("--startup-lr", type=float, default=5.0e-4)
    parser.add_argument(
        "--startup-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enforce-startup-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-expensive-startup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Acknowledge the very large fine-grid cost required by the "
            "theoretical Euler startup refinement."
        ),
    )
    parser.add_argument(
        "--startup-layer-warning-threshold", type=positive_int, default=1000
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mhdbr_euler_startup",
    )
    return parser


def _config(arguments: argparse.Namespace) -> dict[str, Any]:
    values = vars(arguments).copy()
    values.update(
        {
            "K_batch": values.pop("K_batch"),
            "M_branch": values.pop("M_branch"),
        }
    )
    return values


def _write_csv(results: list[TrainingResult], path: Path) -> None:
    rows = [asdict(result) for result in results]
    scalar_keys = [
        "order",
        "seed",
        "y0",
        "absolute_y0_error",
        "z0_rmse",
        "startup_refinement",
        "startup_fine_dt",
        "startup_fine_layers",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_keys})


def main(order: int) -> None:
    arguments = parser_for_order(order).parse_args()
    base_config = _config(arguments)
    if base_config["startup_batch_size"] is None:
        base_config["startup_batch_size"] = base_config["K_batch"]
    if base_config["startup_branches"] is None:
        base_config["startup_branches"] = base_config["M_branch"]
    problem = build_problem(base_config)
    results: list[TrainingResult] = []
    for run_index in range(arguments.runs):
        run_config = base_config.copy()
        run_config["seed"] = arguments.seed + run_index
        result = train_mhdbr(problem, order, run_config)
        results.append(result)
        print(
            f"run={run_index + 1}/{arguments.runs}, seed={result.seed}, "
            f"Y0={result.y0:.9f}, abs-error={result.absolute_y0_error:.3e}, "
            f"Z-RMSE={result.z0_rmse:.3e}, time={result.elapsed_seconds:.1f}s"
        )

    output_directory = arguments.output_root.expanduser().resolve() / f"ex1_j{order}"
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(results, output_directory / "runs.csv")
    y_values = np.asarray([result.y0 for result in results])
    summary = {
        "problem": "Example 1",
        "scheme": f"MHDBR J={order} with fine-grid Euler startup",
        "device": str(DEVICE),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()},
        "mean_y0": float(np.mean(y_values)),
        "std_y0": float(np.std(y_values, ddof=1)) if y_values.size > 1 else 0.0,
        "mae_y0": float(np.mean(np.abs(y_values - problem.true_y0))),
        "runs": [asdict(result) for result in results],
    }
    with (output_directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("\nFinal summary")
    print(
        f"J={order}, trueY0={problem.true_y0:.9f}, "
        f"meanY0={summary['mean_y0']:.9f}, std={summary['std_y0']:.3e}, "
        f"MAE={summary['mae_y0']:.3e}"
    )
    print(f"outputs={output_directory}")
