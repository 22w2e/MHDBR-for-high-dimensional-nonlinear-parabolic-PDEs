#!/usr/bin/env python3
"""Example 3 MHDBR J=1,...,4 with fine-grid Euler startup values.

The nonterminal startup layers required by a multistep formula are obtained
from a one-step backward Euler neural/Monte-Carlo regression on a refined time
grid.  The analytic solution is used only to report errors, never to construct
the startup values.  Use ex3MHDBRJ1Euler.py,...,ex3MHDBRJ4Euler.py as the
direct entry points.  This module deliberately contains no plotting code.
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


def initial_state(dimension: int) -> torch.Tensor:
    """Return (1, 0.5, 1, 0.5, ...) truncated to dimension d."""
    state = torch.empty(dimension, device=DEVICE)
    state[0::2] = 1.0
    state[1::2] = 0.5
    return state


def build_problem(config: dict[str, Any], dimension: int) -> Problem:
    terminal_time = float(config["T"])
    rate = float(config["r"])
    sigma = float(config["sigma"])
    x0 = initial_state(dimension)

    def sample_state(time_value: float, batch_size: int) -> torch.Tensor:
        if abs(time_value) < 1.0e-14:
            return x0.unsqueeze(0).expand(batch_size, -1).clone()
        normal = torch.randn(batch_size, dimension, device=DEVICE)
        return x0.unsqueeze(0) * torch.exp(
            -0.5 * sigma**2 * time_value
            + sigma * math.sqrt(time_value) * normal
        )

    def euler_transition(
        state: torch.Tensor, delta_w: torch.Tensor, delta: float
    ) -> torch.Tensor:
        del delta
        return state.unsqueeze(1) + sigma * state.unsqueeze(1) * delta_w

    def main_transition(
        state: torch.Tensor, cumulative_dw: torch.Tensor, elapsed: float
    ) -> torch.Tensor:
        # This is the same exact conditional GBM transition as in the existing
        # Example-3 codes.  Only the startup solver is replaced by Euler.
        return state.unsqueeze(1) * torch.exp(
            -0.5 * sigma**2 * elapsed + sigma * cumulative_dw
        )

    # The following two functions are prescribed terminal data g and
    # sigma^T Dg.  They are not analytic nonterminal startup values.
    def terminal_y(state: torch.Tensor) -> torch.Tensor:
        return torch.sum(state**2, dim=-1, keepdim=True)

    def terminal_z(state: torch.Tensor) -> torch.Tensor:
        return 2.0 * sigma * state**2

    def generator(
        time_value: float,
        state: torch.Tensor,
        y_value: torch.Tensor,
        z_value: torch.Tensor,
    ) -> torch.Tensor:
        del time_value, state
        return -rate * (
            y_value - torch.sum(z_value, dim=-1, keepdim=True) / sigma
        )

    growth = math.exp((rate + sigma**2) * terminal_time)
    true_y0 = growth * float(torch.sum(x0**2).item())
    true_z0 = 2.0 * sigma * growth * x0**2
    return Problem(
        name=f"Example 3 (d={dimension})",
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
    parser = argparse.ArgumentParser(
        description=(
            f"Example 3 MHDBR J={order} with nonterminal startup layers "
            "computed by refined Euler neural regression."
        )
    )
    parser.add_argument("--d-list", type=positive_int, nargs="+", default=[2, 10, 50, 100])
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--N", type=positive_int, default=20)
    parser.add_argument("--K-batch", type=positive_int, default=64)
    parser.add_argument("--M-branch", type=positive_int, default=256)
    parser.add_argument("--hidden-layers", type=positive_int, default=2)
    parser.add_argument("--hidden-dim-offset", type=positive_int, default=110)
    parser.add_argument("--train-steps", type=positive_int, default=3000)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=3003 + 100 * order)
    parser.add_argument(
        "--warm-start-from-next",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--startup-train-steps", type=positive_int, default=3000)
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


def _write_csv(results: list[tuple[int, TrainingResult]], path: Path) -> None:
    fieldnames = [
        "dimension",
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
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for dimension, result in results:
            row = asdict(result)
            row["dimension"] = dimension
            writer.writerow({key: row[key] for key in fieldnames})


def main(order: int) -> None:
    arguments = parser_for_order(order).parse_args()
    base_config = vars(arguments).copy()
    if base_config["startup_batch_size"] is None:
        base_config["startup_batch_size"] = base_config["K_batch"]
    if base_config["startup_branches"] is None:
        base_config["startup_branches"] = base_config["M_branch"]

    all_results: list[tuple[int, TrainingResult]] = []
    summaries: list[dict[str, Any]] = []
    for dimension in arguments.d_list:
        problem = build_problem(base_config, dimension)
        dimension_results: list[TrainingResult] = []
        for run_index in range(arguments.runs):
            run_config = base_config.copy()
            run_config["seed"] = arguments.seed + run_index
            result = train_mhdbr(problem, order, run_config)
            dimension_results.append(result)
            all_results.append((dimension, result))
            print(
                f"d={dimension}, run={run_index + 1}/{arguments.runs}, "
                f"seed={result.seed}, Y0={result.y0:.9f}, "
                f"abs-error={result.absolute_y0_error:.3e}, "
                f"Z-RMSE={result.z0_rmse:.3e}, "
                f"time={result.elapsed_seconds:.1f}s"
            )

        y_values = np.asarray([result.y0 for result in dimension_results])
        summaries.append(
            {
                "dimension": dimension,
                "true_y0": problem.true_y0,
                "mean_y0": float(np.mean(y_values)),
                "std_y0": (
                    float(np.std(y_values, ddof=1)) if y_values.size > 1 else 0.0
                ),
                "mae_y0": float(np.mean(np.abs(y_values - problem.true_y0))),
            }
        )

    output_directory = arguments.output_root.expanduser().resolve() / f"ex3_j{order}"
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(all_results, output_directory / "runs.csv")
    payload = {
        "problem": "Example 3",
        "scheme": f"MHDBR J={order} with fine-grid Euler startup",
        "device": str(DEVICE),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "summaries": summaries,
        "runs": [
            {"dimension": dimension, **asdict(result)}
            for dimension, result in all_results
        ],
    }
    with (output_directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\nFinal dimension summaries")
    for row in summaries:
        print(
            f"d={row['dimension']}: trueY0={row['true_y0']:.9f}, "
            f"meanY0={row['mean_y0']:.9f}, std={row['std_y0']:.3e}, "
            f"MAE={row['mae_y0']:.3e}"
        )
    print(f"outputs={output_directory}")
