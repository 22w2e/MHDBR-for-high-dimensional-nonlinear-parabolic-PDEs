#!/usr/bin/env python3
"""Example 3: 2022 MDBDP PyTorch/Monte-Carlo dimension experiment.

This self-contained program transfers the MDBDP algorithm used in
``26MDBDPex1.py`` to the
Example-3 Black--Scholes--Barenblatt FBSDE

    dX_t = sigma diag(X_t) dW_t,
    -dY_t = -r (Y_t - sum(Z_t)/sigma) dt - Z_t dW_t,
    Y_T = |X_T|^2.

At time layer i, MDBDP simulates the complete conditional tail path from i to
N and minimizes the terminal reconstruction loss

    g(X_N) - [Y_i - sum_{j=i}^{N-1} f_j h
                    + sum_{j=i}^{N-1} Z_j Delta W_j].

Only the current Y_i/Z_i networks receive gradients; all already-trained
future networks are evaluated under ``torch.no_grad()``, exactly as in the
Example-1 MDBDP implementation.  The diagonal geometric Brownian motion is
advanced with its exact transition, avoiding an unrelated forward Euler error.

Example:

    python examples/example3/run_mdbdp.py --d-list 2 10 50 100 --runs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


# Everything is defined locally: this program does not import, instantiate,
# or load any MHDBR model or checkpoint.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_FINAL = 1.0
RATE = 0.05
SIGMA = 0.4


class FBSDE_Net_Final(nn.Module):
    """The (t, x, previous dW) network used by the original Example-1 MDBDP."""

    def __init__(
        self,
        state_dim: int,
        noise_dim: int,
        hidden_layers: int = 2,
        hidden_dim_offset: int = 110,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        input_dim = 1 + state_dim + noise_dim
        hidden_dim = state_dim + hidden_dim_offset
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        t_value: float | torch.Tensor,
        x_value: torch.Tensor,
        dw_previous: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(t_value, (float, int)):
            t_tensor = torch.full(
                (x_value.shape[0], 1),
                float(t_value),
                device=x_value.device,
                dtype=x_value.dtype,
            )
        elif t_value.ndim == 0:
            t_tensor = torch.ones(
                x_value.shape[0], 1, device=x_value.device, dtype=x_value.dtype
            ) * t_value
        elif t_value.ndim == 1:
            t_tensor = t_value.unsqueeze(1)
        else:
            t_tensor = t_value
        return self.network(torch.cat([t_tensor, x_value, dw_previous], dim=1))


def initial_state(dimension: int) -> torch.Tensor:
    """Return X0=(1, 0.5, 1, 0.5, ...) on the active device."""
    indices = torch.arange(dimension, device=device)
    ones = torch.ones(dimension, device=device)
    halves = torch.full((dimension,), 0.5, device=device)
    return torch.where(indices % 2 == 0, ones, halves)


def analytic_solution(
    t_value: float | torch.Tensor,
    x_value: torch.Tensor,
    rate: float = RATE,
    sigma: float = SIGMA,
    terminal_time: float = T_FINAL,
) -> torch.Tensor:
    """Exact solution, used only to report the numerical error."""
    if isinstance(t_value, torch.Tensor):
        factor = torch.exp((rate + sigma**2) * (terminal_time - t_value))
        if factor.ndim == 1:
            factor = factor.unsqueeze(1)
    else:
        factor = math.exp((rate + sigma**2) * (terminal_time - float(t_value)))
    return factor * torch.sum(x_value**2, dim=-1, keepdim=True)


def analytic_z(
    t_value: float | torch.Tensor,
    x_value: torch.Tensor,
    rate: float = RATE,
    sigma: float = SIGMA,
    terminal_time: float = T_FINAL,
) -> torch.Tensor:
    """Exact Z, used only to report the numerical error."""
    if isinstance(t_value, torch.Tensor):
        factor = torch.exp((rate + sigma**2) * (terminal_time - t_value))
        if factor.ndim == 1:
            factor = factor.unsqueeze(1)
    else:
        factor = math.exp((rate + sigma**2) * (terminal_time - float(t_value)))
    return 2.0 * sigma * factor * x_value**2


def generator_f(
    y_value: torch.Tensor,
    z_value: torch.Tensor,
    rate: float = RATE,
    sigma: float = SIGMA,
) -> torch.Tensor:
    """Example-3 BSDE generator f=-r*(Y-sum(Z)/sigma)."""
    return -rate * (
        y_value - torch.sum(z_value, dim=-1, keepdim=True) / sigma
    )


def exact_sample_x(
    t_value: float,
    batch_size: int,
    dimension: int,
    sigma: float,
    x0: torch.Tensor,
) -> torch.Tensor:
    """Sample X_t exactly from the diagonal geometric Brownian motion."""
    if abs(t_value) < 1.0e-14:
        return x0.unsqueeze(0).expand(batch_size, -1).clone()
    normal = torch.randn(batch_size, dimension, device=device)
    exponent = -0.5 * sigma**2 * t_value + sigma * math.sqrt(t_value) * normal
    return x0.unsqueeze(0) * torch.exp(exponent)


def exact_gbm_step(
    x_value: torch.Tensor,
    delta_w: torch.Tensor,
    dt: float,
    sigma: float,
) -> torch.Tensor:
    """One exact conditional step of dX=sigma*diag(X)dW."""
    return x_value * torch.exp(-0.5 * sigma**2 * dt + sigma * delta_w)


def train_mdbdp(config: dict[str, Any]) -> tuple[
    FBSDE_Net_Final,
    FBSDE_Net_Final,
    dict[str, dict[str, list[float]]],
]:
    """Train the Example-3 MDBDP scheme for one random seed."""
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    np.random.seed(config["seed"])

    dimension = config["d"]
    terminal_time = config.get("T", T_FINAL)
    n_steps = config["N"]
    if n_steps < 1:
        raise ValueError("MDBDP requires N >= 1.")
    batch_size = config["K_batch"]
    branches = config["M_branch"]
    rate = config.get("r", RATE)
    sigma = config.get("sigma", SIGMA)
    dt = terminal_time / n_steps
    x0 = initial_state(dimension)
    true_y0 = float(
        analytic_solution(0.0, x0.unsqueeze(0), rate, sigma, terminal_time).item()
    )

    models_y = [
        FBSDE_Net_Final(
            dimension,
            dimension,
            config["hidden_layers"],
            config["hidden_dim_offset"],
            1,
        ).to(device)
        for _ in range(n_steps)
    ]
    models_z = [
        FBSDE_Net_Final(
            dimension,
            dimension,
            config["hidden_layers"],
            config["hidden_dim_offset"],
            dimension,
        ).to(device)
        for _ in range(n_steps)
    ]
    time_grid = torch.linspace(0.0, terminal_time, n_steps + 1, device=device)
    history = {"step_0": {"loss": [], "rel_err_y0": []}}
    t0 = torch.zeros(1, 1, device=device)
    x0_batch = x0.unsqueeze(0)
    dw0 = torch.zeros(1, dimension, device=device)

    print("\n>>> Training Example-3 MDBDP backward scheme <<<")
    print("tail path = complete i-to-N path; transition = exact GBM")
    for index in reversed(range(n_steps)):
        t_current = time_grid[index].item()
        optimizer = Adam(
            list(models_y[index].parameters()) + list(models_z[index].parameters()),
            lr=config["lr"],
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(
                1,
                int(config["train_steps"] * config.get("lr_decay_fraction", 0.6)),
            ),
            gamma=config.get("lr_decay_gamma", 0.5),
        )

        print(f"  Example-3 MDBDP Step {index} Training...", end="\r")
        for _step in range(config["train_steps"]):
            optimizer.zero_grad()

            # Exact marginal sample of X_i from the alternating initial state.
            x_current = exact_sample_x(
                t_current, batch_size, dimension, sigma, x0
            )

            # Complete conditional tail path: shapes follow 26MDBDPex1.py.
            tail_steps = n_steps - index
            dw_tail = math.sqrt(dt) * torch.randn(
                batch_size,
                branches,
                tail_steps,
                dimension,
                device=device,
            )
            x_path = torch.empty(
                batch_size,
                branches,
                tail_steps + 1,
                dimension,
                device=device,
            )
            x_path[:, :, 0, :] = x_current.unsqueeze(1).expand(
                -1, branches, -1
            )
            for tail_index in range(tail_steps):
                x_path[:, :, tail_index + 1, :] = exact_gbm_step(
                    x_path[:, :, tail_index, :],
                    dw_tail[:, :, tail_index, :],
                    dt,
                    sigma,
                )

            # Current layer is the only part kept in the computation graph.
            dw_zero = torch.zeros(batch_size, dimension, device=device)
            y_current = models_y[index](t_current, x_current, dw_zero)
            z_current = models_z[index](t_current, x_current, dw_zero)
            y_current_expanded = y_current.unsqueeze(1).expand(
                -1, branches, -1
            )
            z_current_expanded = z_current.unsqueeze(1).expand(
                -1, branches, -1
            )
            f_current = generator_f(
                y_current_expanded, z_current_expanded, rate, sigma
            )
            z_dw_current = torch.sum(
                z_current_expanded * dw_tail[:, :, 0, :],
                dim=-1,
                keepdim=True,
            )
            current_part = (
                y_current_expanded - f_current * dt + z_dw_current
            )

            # Previously trained future layers are frozen, as in ex1 MDBDP.
            future_part = torch.zeros(
                batch_size, branches, 1, device=device
            )
            if index < n_steps - 1:
                with torch.no_grad():
                    for tail_index in range(1, tail_steps):
                        future_index = index + tail_index
                        future_time = time_grid[future_index].item()
                        x_future = x_path[:, :, tail_index, :]
                        dw_previous = dw_tail[:, :, tail_index - 1, :]
                        y_future = models_y[future_index](
                            future_time,
                            x_future.reshape(-1, dimension),
                            dw_previous.reshape(-1, dimension),
                        ).view(batch_size, branches, 1)
                        z_future = models_z[future_index](
                            future_time,
                            x_future.reshape(-1, dimension),
                            dw_previous.reshape(-1, dimension),
                        ).view(batch_size, branches, dimension)
                        f_future = generator_f(y_future, z_future, rate, sigma)
                        z_dw_future = torch.sum(
                            z_future * dw_tail[:, :, tail_index, :],
                            dim=-1,
                            keepdim=True,
                        )
                        future_part = future_part + (
                            -f_future * dt + z_dw_future
                        )

            terminal_prediction = current_part + future_part
            x_terminal = x_path[:, :, -1, :]
            terminal_value = torch.sum(
                x_terminal**2, dim=-1, keepdim=True
            )
            loss = torch.mean((terminal_value - terminal_prediction) ** 2)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if index == 0:
                history["step_0"]["loss"].append(float(loss.item()))
                with torch.no_grad():
                    y0_value = models_y[0](t0, x0_batch, dw0).item()
                history["step_0"]["rel_err_y0"].append(
                    abs(y0_value - true_y0) / true_y0
                )

        print(
            f"  Example-3 MDBDP Step {index} Done. "
            f"Final Loss={loss.item():.2e}    "
        )

    return models_y[0], models_z[0], history


def evaluate_initial_models(
    model_y0: FBSDE_Net_Final,
    model_z0: FBSDE_Net_Final,
    dimension: int,
) -> tuple[float, np.ndarray]:
    model_y0.eval()
    model_z0.eval()
    x0 = initial_state(dimension).unsqueeze(0)
    t0 = torch.zeros(1, 1, device=device)
    dw0 = torch.zeros(1, dimension, device=device)
    with torch.no_grad():
        y0 = float(model_y0(t0, x0, dw0).item())
        z0 = model_z0(t0, x0, dw0).detach().cpu().numpy().reshape(-1)
    return y0, z0


@dataclass(frozen=True)
class DimensionRun:
    dimension: int
    run_index: int
    seed: int
    true_y0: float
    predicted_y0: float
    absolute_y0_error: float
    relative_y0_error: float
    z0_rmse: float
    z0_relative_l2_error: float
    elapsed_seconds: float


@dataclass(frozen=True)
class DimensionSummary:
    dimension: int
    runs: int
    true_y0: float
    mean_y0: float
    std_y0: float
    y0_mae: float
    y0_relative_error_of_mean: float
    mean_z0_rmse: float
    mean_z0_relative_l2_error: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Example-3 2022 MDBDP experiment over several dimensions, "
            "compatible with the MHDBR dimension-test outputs."
        )
    )
    parser.add_argument(
        "--d-list", type=positive_int, nargs="+", default=[2, 10, 50, 100]
    )
    parser.add_argument("--n-steps", type=positive_int, default=20)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--k-batch", type=positive_int, default=64)
    parser.add_argument("--m-branch", type=positive_int, default=256)
    parser.add_argument("--train-steps", type=positive_int, default=3000)
    parser.add_argument("--hidden-layers", type=positive_int, default=2)
    parser.add_argument("--hidden-dim-offset", type=positive_int, default=110)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--lr-decay-fraction", type=float, default=0.6)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    parser.add_argument("--base-seed", type=int, default=3003)
    parser.add_argument("--torch-threads", type=positive_int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--tag", type=str, default="dimension-test")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mdbdp_dimensions",
    )
    return parser.parse_args()


def write_csv_file(records: list[Any], path: Path, record_type: type[Any]) -> None:
    fieldnames = (
        list(asdict(records[0]).keys()) if records else list(record_type.__annotations__)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def main() -> None:
    arguments = parse_arguments()
    if arguments.lr <= 0.0:
        raise ValueError("--lr must be positive.")
    if not 0.0 < arguments.lr_decay_fraction <= 1.0:
        raise ValueError("--lr-decay-fraction must lie in (0, 1].")
    if arguments.lr_decay_gamma <= 0.0:
        raise ValueError("--lr-decay-gamma must be positive.")
    if len(set(arguments.d_list)) != len(arguments.d_list):
        raise ValueError("--d-list must not contain duplicate dimensions.")
    torch.set_default_dtype(
        torch.float64 if arguments.dtype == "float64" else torch.float32
    )
    torch.set_num_threads(arguments.torch_threads)

    output_directory = arguments.output_root / arguments.tag
    output_directory.mkdir(parents=True, exist_ok=True)
    all_runs: list[DimensionRun] = []
    summaries: list[DimensionSummary] = []

    print("\nExample-3 MDBDP dimension experiment")
    print(
        f"torch={torch.__version__}, dtype={torch.get_default_dtype()}, "
        f"device={device}, dimensions={arguments.d_list}"
    )
    for dimension in arguments.d_list:
        x0 = initial_state(dimension).unsqueeze(0)
        true_y0 = float(analytic_solution(0.0, x0).item())
        true_z0 = analytic_z(0.0, x0).detach().cpu().numpy().reshape(-1)
        dimension_runs: list[DimensionRun] = []
        print(f"\n=== d={dimension}, exact Y0={true_y0:.10f} ===")

        for run_index in range(arguments.runs):
            seed = arguments.base_seed + run_index
            config = {
                "d": dimension,
                "T": T_FINAL,
                "r": RATE,
                "sigma": SIGMA,
                "N": arguments.n_steps,
                "K_batch": arguments.k_batch,
                "M_branch": arguments.m_branch,
                "hidden_layers": arguments.hidden_layers,
                "hidden_dim_offset": arguments.hidden_dim_offset,
                "train_steps": arguments.train_steps,
                "lr": arguments.lr,
                "lr_decay_fraction": arguments.lr_decay_fraction,
                "lr_decay_gamma": arguments.lr_decay_gamma,
                "seed": seed,
            }
            started = time.perf_counter()
            model_y0, model_z0, _history = train_mdbdp(config)
            predicted_y0, predicted_z0 = evaluate_initial_models(
                model_y0, model_z0, dimension
            )
            elapsed = time.perf_counter() - started
            z_difference = predicted_z0 - true_z0
            record = DimensionRun(
                dimension=dimension,
                run_index=run_index + 1,
                seed=seed,
                true_y0=true_y0,
                predicted_y0=predicted_y0,
                absolute_y0_error=abs(predicted_y0 - true_y0),
                relative_y0_error=abs(predicted_y0 - true_y0) / true_y0,
                z0_rmse=float(np.sqrt(np.mean(z_difference**2))),
                z0_relative_l2_error=float(
                    np.linalg.norm(z_difference) / np.linalg.norm(true_z0)
                ),
                elapsed_seconds=elapsed,
            )
            dimension_runs.append(record)
            all_runs.append(record)
            print(
                f"run={run_index + 1}/{arguments.runs}, seed={seed}, "
                f"Y0={predicted_y0:.8f}, Y-MAE={record.absolute_y0_error:.3e}, "
                f"Z-RMSE={record.z0_rmse:.3e}, time={elapsed:.1f}s"
            )

        y_values = np.asarray([record.predicted_y0 for record in dimension_runs])
        summary = DimensionSummary(
            dimension=dimension,
            runs=len(dimension_runs),
            true_y0=true_y0,
            mean_y0=float(np.mean(y_values)),
            std_y0=(
                float(np.std(y_values, ddof=1)) if y_values.size > 1 else 0.0
            ),
            y0_mae=float(
                np.mean([record.absolute_y0_error for record in dimension_runs])
            ),
            y0_relative_error_of_mean=abs(float(np.mean(y_values)) - true_y0)
            / true_y0,
            mean_z0_rmse=float(
                np.mean([record.z0_rmse for record in dimension_runs])
            ),
            mean_z0_relative_l2_error=float(
                np.mean([record.z0_relative_l2_error for record in dimension_runs])
            ),
        )
        summaries.append(summary)

    write_csv_file(all_runs, output_directory / "dimension_runs.csv", DimensionRun)
    write_csv_file(
        summaries, output_directory / "dimension_summary.csv", DimensionSummary
    )
    result_document = {
        "problem": "Example 3 Black-Scholes-Barenblatt",
        "scheme": "2022 MDBDP complete-tail terminal reconstruction loss",
        "forward_transition": "exact geometric Brownian motion",
        "arguments": {**vars(arguments), "output_root": str(arguments.output_root)},
        "runs": [asdict(record) for record in all_runs],
        "summaries": [asdict(summary) for summary in summaries],
    }
    with (output_directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result_document, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")

    print("\nFinal dimension summaries")
    for summary in summaries:
        print(
            f"d={summary.dimension}: trueY0={summary.true_y0:.8f}, "
            f"meanY0={summary.mean_y0:.8f}, std={summary.std_y0:.3e}, "
            f"Y-MAE={summary.y0_mae:.3e}, "
            f"Y-rel(mean)={summary.y0_relative_error_of_mean:.3e}, "
            f"Z-RMSE={summary.mean_z0_rmse:.3e}"
        )
    print(f"outputs={output_directory}")


if __name__ == "__main__":
    main()
