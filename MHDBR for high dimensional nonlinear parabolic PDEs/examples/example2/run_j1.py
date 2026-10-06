#!/usr/bin/env python3
"""Example 2 MHDBR J=1 without plotting.

This is the same log-domain PyTorch/Monte-Carlo MHDBR J=1 algorithm used by
the original Example-2 program.  Since J=1 needs no nonterminal startup
layer, the prescribed terminal payoff is the only starting value and no
refined Euler-startup computation is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mhdbr.euler_startup import Problem, train_mhdbr


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PUBLISHED_REFERENCES = {50: 17.9743, 100: 21.2988}


@dataclass(frozen=True)
class RunRecord:
    dimension: int
    n_steps: int
    k_batch: int
    m_branch: int
    train_steps: int
    run_index: int
    seed: int
    reference_y0: float
    y0: float
    absolute_error: float
    relative_error: float
    elapsed_seconds: float


@dataclass(frozen=True)
class SummaryRecord:
    dimension: int
    n_steps: int
    k_batch: int
    m_branch: int
    train_steps: int
    runs: int
    reference_y0: float
    mean_y0: float
    std_y0: float
    mae: float
    rmse: float
    relative_error_of_mean: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_problem(config: dict[str, Any]) -> Problem:
    dimension = int(config["d"])
    terminal_time = float(config["T"])
    x0_scalar = float(config["x0_scalar"])
    mu = float(config["mu"])
    sigma = float(config["sigma"])
    strike_1 = float(config["K1"])
    strike_2 = float(config["K2"])
    lending_rate = float(config["R1"])
    borrowing_rate = float(config["R2"])
    initial_log_state = torch.zeros(dimension, device=DEVICE)

    def sample_state(time_value: float, batch_size: int) -> torch.Tensor:
        if abs(time_value) < 1.0e-14:
            return torch.zeros(batch_size, dimension, device=DEVICE)
        normal = torch.randn(batch_size, dimension, device=DEVICE)
        return (
            (mu - 0.5 * sigma**2) * time_value
            + sigma * math.sqrt(time_value) * normal
        )

    def transition(
        state: torch.Tensor, delta_w: torch.Tensor, elapsed: float
    ) -> torch.Tensor:
        # In the logarithmic variable this additive Euler transition is exact.
        return (
            state.unsqueeze(1)
            + (mu - 0.5 * sigma**2) * elapsed
            + sigma * delta_w
        )

    def terminal_y(log_state: torch.Tensor) -> torch.Tensor:
        spot = x0_scalar * torch.exp(log_state)
        maximum = torch.max(spot, dim=-1, keepdim=True).values
        return torch.clamp(maximum - strike_1, min=0.0) - 2.0 * torch.clamp(
            maximum - strike_2, min=0.0
        )

    def terminal_z(log_state: torch.Tensor) -> torch.Tensor:
        # A.e. value sigma^T Dg. Ties and strike kinks have probability zero.
        spot = x0_scalar * torch.exp(log_state)
        maximum, winner = torch.max(spot, dim=-1, keepdim=True)
        slope = torch.where(
            maximum > strike_2,
            -torch.ones_like(maximum),
            torch.where(
                maximum > strike_1,
                torch.ones_like(maximum),
                torch.zeros_like(maximum),
            ),
        )
        result = torch.zeros_like(spot)
        return result.scatter(-1, winner, sigma * maximum * slope)

    def generator(
        time_value: float,
        state: torch.Tensor,
        y_value: torch.Tensor,
        z_value: torch.Tensor,
    ) -> torch.Tensor:
        del time_value, state
        sum_z = torch.sum(z_value, dim=-1, keepdim=True)
        return (
            -lending_rate * y_value
            - ((mu - lending_rate) / sigma) * sum_z
            + (borrowing_rate - lending_rate)
            * torch.clamp(sum_z / sigma - y_value, min=0.0)
        )

    return Problem(
        name=f"Example 2 (d={dimension})",
        dimension=dimension,
        terminal_time=terminal_time,
        device=DEVICE,
        initial_state=initial_log_state,
        sample_state=sample_state,
        euler_transition=transition,
        main_transition=transition,
        terminal_y=terminal_y,
        terminal_z=terminal_z,
        generator=generator,
        true_y0=float(config["reference_y0"]),
        true_z0=None,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 2 log-domain MHDBR J=1 (no figures)."
    )
    parser.add_argument("--d", type=positive_int, default=50)
    parser.add_argument("--reference-y0", type=float, default=None)
    parser.add_argument("--T", type=positive_float, default=0.5)
    parser.add_argument("--x0", type=positive_float, default=100.0)
    parser.add_argument("--mu", type=float, default=0.06)
    parser.add_argument("--sigma", type=positive_float, default=0.2)
    parser.add_argument("--K1", type=positive_float, default=120.0)
    parser.add_argument("--K2", type=positive_float, default=150.0)
    parser.add_argument("--R1", type=float, default=0.04)
    parser.add_argument("--R2", type=float, default=0.06)
    parser.add_argument("--N", type=positive_int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--K-batch", type=positive_int, nargs="+", default=[64])
    parser.add_argument("--M-branch", type=positive_int, nargs="+", default=[256])
    parser.add_argument(
        "--train-steps", type=positive_int, nargs="+", default=[6000]
    )
    parser.add_argument(
        "--schedule-mode",
        choices=("paired", "cartesian"),
        default="paired",
        help=(
            "paired broadcasts singleton lists and zips the remaining lists; "
            "cartesian runs all N/K/M/train-step combinations"
        ),
    )
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--hidden-layers", type=positive_int, default=2)
    parser.add_argument("--hidden-dim-offset", type=positive_int, default=110)
    parser.add_argument("--lr", type=positive_float, default=5.0e-4)
    parser.add_argument("--lr-decay-fraction", type=positive_float, default=0.6)
    parser.add_argument("--lr-decay-gamma", type=positive_float, default=0.5)
    parser.add_argument(
        "--warm-start-from-next",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=2002)
    parser.add_argument("--torch-threads", type=positive_int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching completed seed results from results.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mhdbr_j1_no_plot",
    )
    return parser.parse_args()


def broadcast(values: list[int], count: int, name: str) -> list[int]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must contain one value or {count} values")
    return values


def schedules(arguments: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    if arguments.schedule_mode == "cartesian":
        result = list(
            product(
                arguments.N,
                arguments.K_batch,
                arguments.M_branch,
                arguments.train_steps,
            )
        )
        if len(set(result)) != len(result):
            raise ValueError("duplicate numerical configurations were requested")
        return result
    count = max(
        len(arguments.N),
        len(arguments.K_batch),
        len(arguments.M_branch),
        len(arguments.train_steps),
    )
    result = list(
        zip(
            broadcast(arguments.N, count, "--N"),
            broadcast(arguments.K_batch, count, "--K-batch"),
            broadcast(arguments.M_branch, count, "--M-branch"),
            broadcast(arguments.train_steps, count, "--train-steps"),
        )
    )
    if len(set(result)) != len(result):
        raise ValueError("duplicate numerical configurations were requested")
    return result


def reference_value(arguments: argparse.Namespace) -> float:
    if arguments.reference_y0 is not None:
        if not math.isfinite(arguments.reference_y0):
            raise ValueError("--reference-y0 must be finite")
        return float(arguments.reference_y0)
    standard_physical_parameters = (
        math.isclose(arguments.T, 0.5)
        and math.isclose(arguments.x0, 100.0)
        and math.isclose(arguments.mu, 0.06)
        and math.isclose(arguments.sigma, 0.2)
        and math.isclose(arguments.K1, 120.0)
        and math.isclose(arguments.K2, 150.0)
        and math.isclose(arguments.R1, 0.04)
        and math.isclose(arguments.R2, 0.06)
    )
    if standard_physical_parameters and arguments.d in PUBLISHED_REFERENCES:
        return PUBLISHED_REFERENCES[arguments.d]
    raise ValueError(
        "No automatic reference is available for this problem. "
        "Supply its matching value with --reference-y0."
    )


def write_csv(rows: list[Any], path: Path, row_type: type[Any]) -> None:
    fields = list(row_type.__annotations__)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def experiment_signature(
    arguments: argparse.Namespace, reference_y0: float
) -> dict[str, Any]:
    """Settings that must match before a completed seed may be reused."""
    return {
        "algorithm_revision": "ex2-mhdbr-j1-log-no-plot-v1",
        "problem": {
            "d": arguments.d,
            "T": arguments.T,
            "x0": arguments.x0,
            "mu": arguments.mu,
            "sigma": arguments.sigma,
            "K1": arguments.K1,
            "K2": arguments.K2,
            "R1": arguments.R1,
            "R2": arguments.R2,
            "reference_y0": reference_y0,
        },
        "network_and_optimizer": {
            "hidden_layers": arguments.hidden_layers,
            "hidden_dim_offset": arguments.hidden_dim_offset,
            "lr": arguments.lr,
            "lr_decay_fraction": arguments.lr_decay_fraction,
            "lr_decay_gamma": arguments.lr_decay_gamma,
            "warm_start_from_next": arguments.warm_start_from_next,
            "dtype": arguments.dtype,
            "torch_threads": arguments.torch_threads,
        },
        "base_seed": arguments.seed,
    }


def load_existing_runs(
    output_root: Path, signature: dict[str, Any], enabled: bool
) -> dict[tuple[int, int, int, int, int], RunRecord]:
    if not enabled:
        return {}
    result_path = output_root / "results.json"
    if not result_path.exists():
        return {}
    try:
        with result_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        print(f"existing results could not be read ({error}); reuse disabled")
        return {}
    if payload.get("experiment_signature") != signature:
        print("existing parameters differ; no previous seed result is reused")
        return {}
    existing: dict[tuple[int, int, int, int, int], RunRecord] = {}
    for raw in payload.get("runs", []):
        try:
            record = RunRecord(**raw)
        except TypeError:
            print("existing result schema differs; reuse disabled")
            return {}
        key = (
            record.n_steps,
            record.k_batch,
            record.m_branch,
            record.train_steps,
            record.seed,
        )
        existing[key] = record
    return existing


def save_results(
    arguments: argparse.Namespace,
    signature: dict[str, Any],
    runs: list[RunRecord],
    summaries: list[SummaryRecord],
    output_root: Path,
    completed: bool,
) -> None:
    write_csv(runs, output_root / "runs.csv", RunRecord)
    write_csv(summaries, output_root / "summary.csv", SummaryRecord)
    payload = {
        "completed": completed,
        "problem": "Example 2 dual-rate financial model",
        "scheme": "MHDBR J=1 in the centered logarithmic state",
        "device": str(DEVICE),
        "experiment_signature": signature,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "runs": [asdict(row) for row in runs],
        "summaries": [asdict(row) for row in summaries],
    }
    temporary = output_root / "results.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(output_root / "results.json")


def main() -> None:
    arguments = parse_arguments()
    if arguments.K2 <= arguments.K1:
        raise ValueError("K2 must be greater than K1")
    if arguments.R2 < arguments.R1:
        raise ValueError("R2 must be at least R1")
    if not 0.0 < arguments.lr_decay_fraction <= 1.0:
        raise ValueError("--lr-decay-fraction must lie in (0,1]")
    torch.set_default_dtype(
        torch.float64 if arguments.dtype == "float64" else torch.float32
    )
    torch.set_num_threads(arguments.torch_threads)
    reference_y0 = reference_value(arguments)
    output_root = arguments.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(arguments, reference_y0)
    existing = load_existing_runs(output_root, signature, arguments.resume)

    all_runs: list[RunRecord] = []
    all_summaries: list[SummaryRecord] = []
    for n_steps, k_batch, m_branch, train_steps in schedules(arguments):
        config = {
            "d": arguments.d,
            "T": arguments.T,
            "x0_scalar": arguments.x0,
            "mu": arguments.mu,
            "sigma": arguments.sigma,
            "K1": arguments.K1,
            "K2": arguments.K2,
            "R1": arguments.R1,
            "R2": arguments.R2,
            "reference_y0": reference_y0,
            "N": n_steps,
            "K_batch": k_batch,
            "M_branch": m_branch,
            "hidden_layers": arguments.hidden_layers,
            "hidden_dim_offset": arguments.hidden_dim_offset,
            "train_steps": train_steps,
            "lr": arguments.lr,
            "lr_decay_fraction": arguments.lr_decay_fraction,
            "lr_decay_gamma": arguments.lr_decay_gamma,
            "warm_start_from_next": arguments.warm_start_from_next,
        }
        problem = build_problem(config)
        configuration_runs: list[RunRecord] = []
        for run_index in range(1, arguments.runs + 1):
            run_config = config.copy()
            run_config["seed"] = arguments.seed + run_index - 1
            key = (
                n_steps,
                k_batch,
                m_branch,
                train_steps,
                run_config["seed"],
            )
            if key in existing:
                previous = existing[key]
                record = RunRecord(**{**asdict(previous), "run_index": run_index})
                print(
                    f"reused N={n_steps}, seed={record.seed} from results.json"
                )
            else:
                result = train_mhdbr(problem, 1, run_config)
                record = RunRecord(
                    dimension=arguments.d,
                    n_steps=n_steps,
                    k_batch=k_batch,
                    m_branch=m_branch,
                    train_steps=train_steps,
                    run_index=run_index,
                    seed=result.seed,
                    reference_y0=reference_y0,
                    y0=result.y0,
                    absolute_error=result.absolute_y0_error,
                    relative_error=result.absolute_y0_error / abs(reference_y0),
                    elapsed_seconds=result.elapsed_seconds,
                )
            configuration_runs.append(record)
            all_runs.append(record)
            print(
                f"N={n_steps}, run={run_index}/{arguments.runs}, "
                f"seed={record.seed}, Y0={record.y0:.8f}, "
                f"abs-error={record.absolute_error:.3e}"
            )
            save_results(
                arguments,
                signature,
                all_runs,
                all_summaries,
                output_root,
                completed=False,
            )

        values = np.asarray([row.y0 for row in configuration_runs])
        errors = values - reference_y0
        summary = SummaryRecord(
            dimension=arguments.d,
            n_steps=n_steps,
            k_batch=k_batch,
            m_branch=m_branch,
            train_steps=train_steps,
            runs=len(configuration_runs),
            reference_y0=reference_y0,
            mean_y0=float(np.mean(values)),
            std_y0=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors**2))),
            relative_error_of_mean=(
                abs(float(np.mean(values)) - reference_y0) / abs(reference_y0)
            ),
        )
        all_summaries.append(summary)
        save_results(
            arguments,
            signature,
            all_runs,
            all_summaries,
            output_root,
            completed=False,
        )

    save_results(
        arguments,
        signature,
        all_runs,
        all_summaries,
        output_root,
        completed=True,
    )

    print("\nFinal summaries")
    for row in all_summaries:
        print(
            f"d={row.dimension}, N={row.n_steps}, K={row.k_batch}, "
            f"M={row.m_branch}: reference={row.reference_y0:.8f}, "
            f"mean={row.mean_y0:.8f}, std={row.std_y0:.3e}, "
            f"MAE={row.mae:.3e}, RMSE={row.rmse:.3e}, "
            f"RelErr(mean)={row.relative_error_of_mean:.4%}"
        )
    print(f"outputs={output_root}")


if __name__ == "__main__":
    main()
