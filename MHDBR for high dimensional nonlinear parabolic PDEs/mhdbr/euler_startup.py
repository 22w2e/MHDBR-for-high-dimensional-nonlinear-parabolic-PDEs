#!/usr/bin/env python3
"""Shared PyTorch/Monte-Carlo MHDBR trainer with fine-grid Euler startup.

This module contains no plotting code.  For an order-J main MHDBR scheme,
J-1 coarse startup layers are computed by a one-step backward Euler regression
on a refined grid.  If the Euler strong error is O(delta**0.5), matching the
O(h**J) main-scheme error requires delta**0.5 <= c*h**J, hence a refinement
of order h**(1-2J) inside every coarse interval.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


Tensor = torch.Tensor


class FBSDE_Net(nn.Module):
    """Same Tanh network architecture used by the original examples."""

    def __init__(
        self,
        state_dim: int,
        hidden_layers: int,
        hidden_dim_offset: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        input_dim = 1 + 2 * state_dim
        hidden_dim = state_dim + hidden_dim_offset
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        time_value: float | Tensor,
        state: Tensor,
        previous_dw: Tensor | None = None,
    ) -> Tensor:
        if previous_dw is None:
            previous_dw = torch.zeros_like(state)
        if isinstance(time_value, (float, int)):
            time_tensor = torch.full(
                (state.shape[0], 1),
                float(time_value),
                device=state.device,
                dtype=state.dtype,
            )
        elif time_value.ndim == 0:
            time_tensor = torch.ones(
                state.shape[0], 1, device=state.device, dtype=state.dtype
            ) * time_value
        elif time_value.ndim == 1:
            time_tensor = time_value.unsqueeze(1)
        else:
            time_tensor = time_value
        return self.network(torch.cat([time_tensor, state, previous_dw], dim=1))


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    terminal_time: float
    device: torch.device
    initial_state: Tensor
    sample_state: Callable[[float, int], Tensor]
    euler_transition: Callable[[Tensor, Tensor, float], Tensor]
    main_transition: Callable[[Tensor, Tensor, float], Tensor]
    terminal_y: Callable[[Tensor], Tensor]
    terminal_z: Callable[[Tensor], Tensor]
    generator: Callable[[float, Tensor, Tensor, Tensor], Tensor]
    true_y0: float
    true_z0: Tensor | None


@dataclass
class TrainingResult:
    order: int
    seed: int
    y0: float
    z0: list[float]
    absolute_y0_error: float
    z0_rmse: float | None
    startup_refinement: int
    startup_fine_dt: float | None
    startup_fine_layers: int
    elapsed_seconds: float
    final_main_loss_y: dict[int, float]
    final_main_loss_z: dict[int, float]
    final_startup_loss_y: dict[int, float]
    final_startup_loss_z: dict[int, float]


def mhdbr_coefficients(
    order: int, dt: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return the coefficients used by the existing J=1,2,3,4 codes."""
    alpha_values = {
        1: [0.5, 0.5],
        2: [5.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0],
        3: [3.0 / 8.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0],
        4: [
            251.0 / 720.0,
            323.0 / 360.0,
            -11.0 / 30.0,
            53.0 / 360.0,
            -19.0 / 720.0,
        ],
    }
    beta_gamma_values = {
        1: [-1.0, 1.0],
        2: [-3.0 / 2.0, 2.0, -1.0 / 2.0],
        3: [-11.0 / 6.0, 3.0, -3.0 / 2.0, 1.0 / 3.0],
        4: [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0],
    }
    if order not in alpha_values:
        raise ValueError("order must be one of 1, 2, 3, 4")
    alpha = torch.tensor(alpha_values[order], device=device)
    beta = torch.tensor(beta_gamma_values[order], device=device) / dt
    return alpha, beta


def theoretical_startup_refinement(
    coarse_dt: float,
    order: int,
    target_factor: float,
) -> int:
    """Smallest R such that sqrt(coarse_dt/R) <= target_factor*h**J."""
    if order == 1:
        return 1
    if target_factor <= 0.0:
        raise ValueError("startup_target_factor must be positive")
    target = target_factor * coarse_dt**order
    return max(1, int(math.ceil(coarse_dt / (target * target))))


def _new_pair(problem: Problem, config: dict[str, Any]) -> tuple[FBSDE_Net, FBSDE_Net]:
    model_y = FBSDE_Net(
        problem.dimension,
        config["hidden_layers"],
        config["hidden_dim_offset"],
        1,
    ).to(problem.device)
    model_z = FBSDE_Net(
        problem.dimension,
        config["hidden_layers"],
        config["hidden_dim_offset"],
        problem.dimension,
    ).to(problem.device)
    return model_y, model_z


def _evaluate_model_pair(
    model_y: FBSDE_Net,
    model_z: FBSDE_Net,
    time_value: float,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    zero = torch.zeros_like(state)
    return model_y(time_value, state, zero), model_z(time_value, state, zero)


def _scheduler(optimizer: Adam, steps: int, config: dict[str, Any]):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, int(steps * config.get("lr_decay_fraction", 0.6))),
        gamma=config.get("lr_decay_gamma", 0.5),
    )


def train_euler_startup(
    problem: Problem,
    order: int,
    n_steps: int,
    config: dict[str, Any],
) -> tuple[
    dict[int, FBSDE_Net],
    dict[int, FBSDE_Net],
    int,
    float | None,
    dict[int, float],
    dict[int, float],
]:
    """Train the J-1 coarse startup layers on a refined Euler grid.

    The terminal layer is always the prescribed terminal condition and is not
    called an "analytic startup".  For J=1 there is no nonterminal startup
    layer, so this function returns empty dictionaries.
    """
    if order == 1:
        return {}, {}, 1, None, {}, {}

    coarse_dt = problem.terminal_time / n_steps
    auto_refinement = theoretical_startup_refinement(
        coarse_dt, order, config["startup_target_factor"]
    )
    requested = config.get("startup_refinement")
    refinement = auto_refinement if requested is None else int(requested)
    if refinement <= 0:
        raise ValueError("startup_refinement must be positive")

    fine_dt = coarse_dt / refinement
    target_error = config["startup_target_factor"] * coarse_dt**order
    if math.sqrt(fine_dt) > target_error * (1.0 + 1.0e-12):
        message = (
            "The requested startup refinement does not satisfy "
            "sqrt(delta) <= startup_target_factor*h**J: "
            f"sqrt(delta)={math.sqrt(fine_dt):.3e}, "
            f"target={target_error:.3e}, required R>={auto_refinement}."
        )
        if config.get("enforce_startup_order", True):
            raise ValueError(message)
        print("WARNING:", message)

    total_fine_layers = (order - 1) * refinement
    if (
        total_fine_layers > config.get("startup_layer_warning_threshold", 10000)
        and not config.get("allow_expensive_startup", False)
    ):
        raise RuntimeError(
            "The theoretically resolved Euler startup needs "
            f"{total_fine_layers} trained fine layers (R={refinement}). "
            "This is the unavoidable cost of matching an O(h^J) method with "
            "an O(delta^0.5) startup method. Re-run with "
            "--allow-expensive-startup, or specify a smaller "
            "--startup-refinement together with --no-enforce-startup-order."
        )

    start_time = problem.terminal_time - (order - 1) * coarse_dt
    startup_y: dict[int, FBSDE_Net] = {}
    startup_z: dict[int, FBSDE_Net] = {}
    final_loss_y: dict[int, float] = {}
    final_loss_z: dict[int, float] = {}
    next_y: FBSDE_Net | None = None
    next_z: FBSDE_Net | None = None

    batch_size = config.get("startup_batch_size", config["K_batch"])
    branches = config.get("startup_branches", config["M_branch"])
    train_steps = config["startup_train_steps"]

    print("\n>>> Fine-grid Euler startup <<<")
    print(
        f"order={order}, coarse h={coarse_dt:.6e}, R={refinement}, "
        f"delta={fine_dt:.6e}, sqrt(delta)={math.sqrt(fine_dt):.6e}, "
        f"h^J={coarse_dt**order:.6e}, fine layers={total_fine_layers}"
    )

    for fine_index in reversed(range(total_fine_layers)):
        current_time = start_time + fine_index * fine_dt
        next_time = current_time + fine_dt
        model_y, model_z = _new_pair(problem, config)
        if config.get("startup_warm_start", True) and next_y is not None:
            model_y.load_state_dict(next_y.state_dict())
            model_z.load_state_dict(next_z.state_dict())

        optimizer_z = Adam(model_z.parameters(), lr=config["startup_lr"])
        optimizer_y = Adam(model_y.parameters(), lr=config["startup_lr"])
        scheduler_z = _scheduler(optimizer_z, train_steps, config)
        scheduler_y = _scheduler(optimizer_y, train_steps, config)

        last_loss_y = math.nan
        last_loss_z = math.nan
        for _ in range(train_steps):
            state = problem.sample_state(current_time, batch_size)
            zero = torch.zeros_like(state)

            # Z_q = E_q[Y_{q+1} Delta W_q] / delta.
            optimizer_z.zero_grad()
            delta_w = math.sqrt(fine_dt) * torch.randn(
                batch_size, branches, problem.dimension, device=problem.device
            )
            state_next = problem.euler_transition(state, delta_w, fine_dt)
            flat_next = state_next.reshape(-1, problem.dimension)
            with torch.no_grad():
                if fine_index + 1 == total_fine_layers:
                    y_next = problem.terminal_y(flat_next)
                else:
                    assert next_y is not None
                    y_next, _ = _evaluate_model_pair(
                        next_y, next_z, next_time, flat_next  # type: ignore[arg-type]
                    )
                y_next = y_next.reshape(batch_size, branches, 1)
                centered_y = y_next - torch.mean(y_next, dim=1, keepdim=True)
                target_z = torch.mean(
                    centered_y * delta_w / fine_dt, dim=1
                )
            prediction_z = model_z(current_time, state, zero)
            loss_z = torch.mean((prediction_z - target_z) ** 2)
            loss_z.backward()
            optimizer_z.step()
            scheduler_z.step()

            # Implicit one-step backward Euler regression for Y_q.
            optimizer_y.zero_grad()
            delta_w_y = math.sqrt(fine_dt) * torch.randn(
                batch_size, branches, problem.dimension, device=problem.device
            )
            state_next_y = problem.euler_transition(state, delta_w_y, fine_dt)
            flat_next_y = state_next_y.reshape(-1, problem.dimension)
            with torch.no_grad():
                if fine_index + 1 == total_fine_layers:
                    y_next_y = problem.terminal_y(flat_next_y)
                else:
                    assert next_y is not None and next_z is not None
                    y_next_y, _ = _evaluate_model_pair(
                        next_y, next_z, next_time, flat_next_y
                    )
                y_next_y = y_next_y.reshape(batch_size, branches, 1)
                fixed_z = model_z(current_time, state, zero).detach()
            prediction_y = model_y(current_time, state, zero)
            driver = problem.generator(
                current_time, state, prediction_y, fixed_z
            )
            target_y = torch.mean(y_next_y, dim=1) + fine_dt * driver
            loss_y = torch.mean((prediction_y - target_y) ** 2)
            loss_y.backward()
            optimizer_y.step()
            scheduler_y.step()
            last_loss_y = float(loss_y.item())
            last_loss_z = float(loss_z.item())

        next_y, next_z = model_y, model_z
        final_loss_y[fine_index] = last_loss_y
        final_loss_z[fine_index] = last_loss_z
        if fine_index % refinement == 0:
            coarse_index = n_steps - order + 1 + fine_index // refinement
            startup_y[coarse_index] = model_y
            startup_z[coarse_index] = model_z
            print(
                f"  stored Euler startup level {coarse_index}: "
                f"LossY={last_loss_y:.3e}, LossZ={last_loss_z:.3e}"
            )

    expected = set(range(n_steps - order + 1, n_steps))
    if set(startup_y) != expected or set(startup_z) != expected:
        raise RuntimeError(
            f"startup levels mismatch: expected {sorted(expected)}, "
            f"got {sorted(startup_y)}"
        )
    return (
        startup_y,
        startup_z,
        refinement,
        fine_dt,
        final_loss_y,
        final_loss_z,
    )


def train_mhdbr(
    problem: Problem,
    order: int,
    config: dict[str, Any],
) -> TrainingResult:
    """Train one MHDBR run with Euler-computed startup layers."""
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started = time.time()
    n_steps = int(config["N"])
    if n_steps < order:
        raise ValueError(f"MHDBR J={order} requires N >= {order}")
    dt = problem.terminal_time / n_steps
    alpha, beta = mhdbr_coefficients(order, dt, problem.device)

    (
        startup_y,
        startup_z,
        refinement,
        fine_dt,
        startup_loss_y,
        startup_loss_z,
    ) = train_euler_startup(problem, order, n_steps, config)

    models_y: dict[int, FBSDE_Net] = {}
    models_z: dict[int, FBSDE_Net] = {}
    final_loss_y: dict[int, float] = {}
    final_loss_z: dict[int, float] = {}
    batch_size = config["K_batch"]
    branches = config["M_branch"]

    def evaluate_future(
        index: int, time_value: float, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        if index == n_steps:
            return problem.terminal_y(state), problem.terminal_z(state)
        if index in startup_y:
            return _evaluate_model_pair(
                startup_y[index], startup_z[index], time_value, state
            )
        if index not in models_y:
            raise RuntimeError(f"future layer {index} has not been trained")
        return _evaluate_model_pair(
            models_y[index], models_z[index], time_value, state
        )

    print(f"\n>>> {problem.name}: MHDBR J={order} main backward scheme <<<")
    print("alpha =", alpha.detach().cpu().numpy())
    print("beta  =", beta.detach().cpu().numpy())
    if order == 1:
        print("startup = terminal condition only; no nonterminal startup layer")
    else:
        print("startup = fine-grid Euler neural regression (no analytic Y/Z startup)")

    # Main J-step scheme computes i=N-J,...,0.
    for index in reversed(range(n_steps - order + 1)):
        current_time = index * dt
        model_y, model_z = _new_pair(problem, config)
        if config.get("warm_start_from_next", False) and index + 1 in models_y:
            model_y.load_state_dict(models_y[index + 1].state_dict())
            model_z.load_state_dict(models_z[index + 1].state_dict())

        layer_steps = int(config["train_steps"])
        optimizer_z = Adam(model_z.parameters(), lr=config["lr"])
        optimizer_y = Adam(model_y.parameters(), lr=config["lr"])
        scheduler_z = _scheduler(optimizer_z, layer_steps, config)
        scheduler_y = _scheduler(optimizer_y, layer_steps, config)
        last_loss_y = math.nan
        last_loss_z = math.nan

        for _ in range(layer_steps):
            state = problem.sample_state(current_time, batch_size)
            zero = torch.zeros_like(state)

            # Z regression with independent conditional branches.
            optimizer_z.zero_grad()
            dw_steps = math.sqrt(dt) * torch.randn(
                batch_size,
                branches,
                order,
                problem.dimension,
                device=problem.device,
            )
            cumulative_dw = torch.cumsum(dw_steps, dim=2)
            target_z = torch.zeros(
                batch_size, problem.dimension, device=problem.device
            )
            for j in range(1, order + 1):
                future_index = index + j
                future_time = future_index * dt
                dw_ij = cumulative_dw[:, :, j - 1, :]
                state_future = problem.main_transition(state, dw_ij, j * dt)
                with torch.no_grad():
                    y_future, _ = evaluate_future(
                        future_index,
                        future_time,
                        state_future.reshape(-1, problem.dimension),
                    )
                    y_future = y_future.reshape(batch_size, branches, 1)
                    centered_y = y_future - torch.mean(
                        y_future, dim=1, keepdim=True
                    )
                target_z = target_z + beta[j] * torch.mean(
                    centered_y * dw_ij, dim=1
                )
            prediction_z = model_z(current_time, state, zero)
            loss_z = torch.mean((prediction_z - target_z) ** 2)
            loss_z.backward()
            optimizer_z.step()
            scheduler_z.step()

            # Y regression with a fresh conditional sample family.
            optimizer_y.zero_grad()
            dw_steps_y = math.sqrt(dt) * torch.randn(
                batch_size,
                branches,
                order,
                problem.dimension,
                device=problem.device,
            )
            cumulative_dw_y = torch.cumsum(dw_steps_y, dim=2)
            prediction_y = model_y(current_time, state, zero)
            with torch.no_grad():
                fixed_z = model_z(current_time, state, zero).detach()
            driver_sum = alpha[0] * problem.generator(
                current_time, state, prediction_y, fixed_z
            ).unsqueeze(1).expand(-1, branches, -1)
            y_ip1: Tensor | None = None
            for j in range(1, order + 1):
                future_index = index + j
                future_time = future_index * dt
                dw_ij = cumulative_dw_y[:, :, j - 1, :]
                state_future = problem.main_transition(state, dw_ij, j * dt)
                flat_future = state_future.reshape(-1, problem.dimension)
                with torch.no_grad():
                    y_future, z_future = evaluate_future(
                        future_index, future_time, flat_future
                    )
                    y_future = y_future.reshape(batch_size, branches, 1)
                    z_future = z_future.reshape(
                        batch_size, branches, problem.dimension
                    )
                if j == 1:
                    y_ip1 = y_future
                driver_sum = driver_sum + alpha[j] * problem.generator(
                    future_time, state_future, y_future, z_future
                )
            assert y_ip1 is not None
            target_y = torch.mean(y_ip1 + dt * driver_sum, dim=1)
            loss_y = torch.mean((prediction_y - target_y) ** 2)
            loss_y.backward()
            optimizer_y.step()
            scheduler_y.step()
            last_loss_y = float(loss_y.item())
            last_loss_z = float(loss_z.item())

        models_y[index] = model_y
        models_z[index] = model_z
        final_loss_y[index] = last_loss_y
        final_loss_z[index] = last_loss_z
        print(
            f"  main level {index} done: "
            f"LossY={last_loss_y:.3e}, LossZ={last_loss_z:.3e}"
        )

    zero_state_noise = torch.zeros_like(problem.initial_state).unsqueeze(0)
    x0 = problem.initial_state.unsqueeze(0)
    with torch.no_grad():
        y0 = float(models_y[0](0.0, x0, zero_state_noise).item())
        z0_tensor = models_z[0](0.0, x0, zero_state_noise).reshape(-1)
    if problem.true_z0 is None:
        z0_rmse = None
    else:
        true_z0 = problem.true_z0.reshape(-1)
        z0_rmse = float(
            torch.sqrt(torch.mean((z0_tensor - true_z0) ** 2)).item()
        )
    return TrainingResult(
        order=order,
        seed=seed,
        y0=y0,
        z0=[float(value) for value in z0_tensor.detach().cpu().tolist()],
        absolute_y0_error=abs(y0 - problem.true_y0),
        z0_rmse=z0_rmse,
        startup_refinement=refinement,
        startup_fine_dt=fine_dt,
        startup_fine_layers=(order - 1) * refinement,
        elapsed_seconds=time.time() - started,
        final_main_loss_y=final_loss_y,
        final_main_loss_z=final_loss_z,
        final_startup_loss_y=startup_loss_y,
        final_startup_loss_z=startup_loss_z,
    )
