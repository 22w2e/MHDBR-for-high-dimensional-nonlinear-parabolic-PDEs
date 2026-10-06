#!/usr/bin/env python3
"""Example 1 with the 2022 MDBDP complete-tail regression method.

At every time layer the algorithm simulates the complete conditional path to
the terminal time and minimizes the terminal reconstruction loss.  This is
the original Example-1 MDBDP algorithm without any plotting dependency.
"""

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 网络定义（保持不变）
# ==========================================
class FBSDE_Net_Final(nn.Module):
    def __init__(self, state_dim, noise_dim, hidden_layers=2, hidden_dim_offset=110, output_dim=1):
        super().__init__()
        self.input_dim = 1 + state_dim + noise_dim
        hidden_dim = state_dim + hidden_dim_offset

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, t, x, dw_prev):
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t = torch.ones(x.shape[0], 1, device=x.device) * t
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        inputs = torch.cat([t, x, dw_prev], dim=1)
        return self.network(inputs)


# ==========================================
# 2. 物理方程
# ==========================================
def analytic_solution(t, X, d):
    sum_x = torch.sum(X, dim=-1, keepdim=True)
    exponent = torch.clamp(t + sum_x / d, -30.0, 30.0)
    return torch.exp(exponent) / (1.0 + torch.exp(exponent))


def get_generator_f(t, X, Y, Z, d, sigma):
    # 支持 (K,d) 或 (K,M,d)
    sum_z = torch.sum(Z, dim=-1, keepdim=True)
    val = (Y - (d + 2) / (2.0 * d)) * (sum_z / sigma)
    return val


# ==========================================
# 3. MDBDP 训练流程
# ==========================================
def train_mdbdp(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    d = config["d"]
    T = config["T"]
    N = config["N"]
    K_batch = config["K_batch"]
    M_branch = config["M_branch"]
    dt = T / N
    sigma = d
    true_val = 0.5

    models_Y = [
        FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], 1).to(device)
        for _ in range(N)
    ]
    models_Z = [
        FBSDE_Net_Final(d, d, config["hidden_layers"], config["hidden_dim_offset"], d).to(device)
        for _ in range(N)
    ]

    t_grid = torch.linspace(0, T, N + 1, device=device)

    loss_history = {
        "step_0": {"loss_y": [], "rel_err": []}
    }

    t0_val = torch.zeros(1, 1, device=device)
    x0_val = torch.zeros(1, d, device=device)
    dw0_val = torch.zeros(1, d, device=device)

    # 反向时间递推
    for i in reversed(range(N)):
        t_curr = t_grid[i].item()

        all_params = list(models_Y[i].parameters()) + list(models_Z[i].parameters())
        optimizer = Adam(all_params, lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(config["train_steps"] * 0.6)),
            gamma=0.5
        )

        print(f"  Step {i} Training...", end="\r")

        for step in range(config["train_steps"]):
            optimizer.zero_grad()

            # -------------------------------------------------
            # 1) 采样当前时刻 X_i
            # 对于本例：X_t = sigma * W_t, x0 = 0
            # 所以 X_i ~ N(0, sigma^2 t_i I_d)
            # -------------------------------------------------
            if i == 0:
                X_i = torch.zeros(K_batch, d, device=device)
            else:
                X_i = sigma * np.sqrt(t_curr) * torch.randn(K_batch, d, device=device)

            # -------------------------------------------------
            # 2) 从当前时刻往后模拟整条尾部路径（MDBDP 核心）
            # dW_tail: (K, M, N-i, d)
            # X_path[:, :, s, :] 表示 t_{i+s} 时刻状态
            # -------------------------------------------------
            n_tail = N - i
            dW_tail = np.sqrt(dt) * torch.randn(K_batch, M_branch, n_tail, d, device=device)

            X_path = torch.zeros(K_batch, M_branch, n_tail + 1, d, device=device)
            X_path[:, :, 0, :] = X_i.unsqueeze(1).expand(-1, M_branch, -1)

            for s in range(n_tail):
                X_path[:, :, s + 1, :] = X_path[:, :, s, :] + sigma * dW_tail[:, :, s, :]

            # -------------------------------------------------
            # 3) 当前时刻 i 的网络预测（这是唯一参与梯度更新的“当前层”网络）
            # -------------------------------------------------
            dummy_dw = torch.zeros(K_batch, d, device=device)

            Y_i = models_Y[i](t_curr, X_i, dummy_dw)              # (K,1)
            Z_i = models_Z[i](t_curr, X_i, dummy_dw)              # (K,d)

            Y_i_exp = Y_i.unsqueeze(1).expand(-1, M_branch, -1)   # (K,M,1)
            Z_i_exp = Z_i.unsqueeze(1).expand(-1, M_branch, -1)   # (K,M,d)

            X_i_exp = X_i.unsqueeze(1).expand(-1, M_branch, -1)   # (K,M,d)

            f_i = get_generator_f(t_curr, X_i_exp, Y_i_exp, Z_i_exp, d, sigma)  # (K,M,1)

            # -------------------------------------------------
            # 4) 按 MDBDP 的多步展开构造终端预测
            #
            # g(X_T) ≈ Y_i - Σ_{j=i}^{N-1} f_j dt + Σ_{j=i}^{N-1} Z_j ΔW_j
            #
            # 所以我们从当前层开始，把 tail 全部加进去
            # -------------------------------------------------
            # 当前层：必须保留计算图
            Z_dW_i = torch.sum(Z_i_exp * dW_tail[:, :, 0, :], dim=-1, keepdim=True)
            current_part = Y_i_exp - f_i * dt + Z_dW_i  # 这一部分需要梯度

            # 未来层：不需要梯度，单独累计
            future_part = torch.zeros(K_batch, M_branch, 1, device=device)

            if i < N - 1:
                with torch.no_grad():
                    for s in range(1, n_tail):
                        j = i + s
                        t_j = t_grid[j].item()

                        X_j = X_path[:, :, s, :]  # (K,M,d)
                        dW_prev_j = dW_tail[:, :, s - 1, :]  # (K,M,d)

                        X_j_flat = X_j.reshape(-1, d)
                        dW_prev_j_flat = dW_prev_j.reshape(-1, d)

                        Y_j = models_Y[j](t_j, X_j_flat, dW_prev_j_flat).view(K_batch, M_branch, 1)
                        Z_j = models_Z[j](t_j, X_j_flat, dW_prev_j_flat).view(K_batch, M_branch, d)

                        f_j = get_generator_f(t_j, X_j, Y_j, Z_j, d, sigma)
                        Z_dW_j = torch.sum(Z_j * dW_tail[:, :, s, :], dim=-1, keepdim=True)

                        future_part = future_part + (-f_j * dt + Z_dW_j)

            # 最终终端预测：在 no_grad 外组合
            terminal_pred = current_part + future_part

            # -------------------------------------------------
            # 6) 终端真值
            # -------------------------------------------------
            X_T = X_path[:, :, -1, :]                      # (K,M,d)
            g_T = analytic_solution(T, X_T, d)            # (K,M,1)

            # -------------------------------------------------
            # 7) MDBDP 多步损失
            # -------------------------------------------------
            loss = torch.mean((g_T - terminal_pred) ** 2)

            # -------------------------------------------------
            # 8) 反向传播
            # -------------------------------------------------
            loss.backward()
            optimizer.step()
            scheduler.step()

            # -------------------------------------------------
            # 9) 记录 step 0 的训练曲线
            # -------------------------------------------------
            if i == 0:
                loss_history["step_0"]["loss_y"].append(loss.item())
                with torch.no_grad():
                    y0_pred = models_Y[0](t0_val, x0_val, dw0_val).item()
                    rel_error = abs(y0_pred - true_val) / true_val
                    loss_history["step_0"]["rel_err"].append(rel_error)

        print(f"  Step {i} Done. Final MDBDP Loss={loss.item():.2e}    ")

    return models_Y[0], loss_history


@dataclass(frozen=True)
class RunRecord:
    run_index: int
    seed: int
    true_y0: float
    predicted_y0: float
    absolute_error: float
    relative_error: float
    final_step0_loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class SummaryRecord:
    dimension: int
    n_steps: int
    k_batch: int
    m_branch: int
    train_steps: int
    runs: int
    true_y0: float
    mean_y0: float
    std_y0: float
    mae: float
    rmse: float
    relative_error_of_mean: float
    mean_elapsed_seconds: float


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example 1 MDBDP experiment without figures."
    )
    parser.add_argument("--d", type=positive_int, default=200)
    parser.add_argument("--T", type=positive_float, default=1.0)
    parser.add_argument("--N", type=positive_int, default=40)
    parser.add_argument("--K-batch", type=positive_int, default=128)
    parser.add_argument("--M-branch", type=positive_int, default=400)
    parser.add_argument("--hidden-layers", type=positive_int, default=2)
    parser.add_argument("--hidden-dim-offset", type=positive_int, default=110)
    parser.add_argument("--train-steps", type=positive_int, default=6000)
    parser.add_argument("--lr", type=positive_float, default=5.0e-4)
    parser.add_argument("--runs", type=positive_int, default=10)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--torch-threads", type=positive_int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "mdbdp",
    )
    return parser.parse_args()


def write_csv(rows: list[object], path: Path, row_type: type[object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row_type.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    arguments = parse_arguments()
    torch.set_default_dtype(
        torch.float64 if arguments.dtype == "float64" else torch.float32
    )
    torch.set_num_threads(arguments.torch_threads)
    output_directory = arguments.output_root.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    true_y0 = 0.5
    records: list[RunRecord] = []

    print("\nExample-1 MDBDP experiment")
    print(
        f"torch={torch.__version__}, dtype={torch.get_default_dtype()}, "
        f"device={device}, d={arguments.d}, N={arguments.N}, "
        f"K={arguments.K_batch}, M={arguments.M_branch}"
    )
    for run_index in range(1, arguments.runs + 1):
        seed = arguments.seed + run_index - 1
        config = {
            "d": arguments.d,
            "T": arguments.T,
            "N": arguments.N,
            "K_batch": arguments.K_batch,
            "M_branch": arguments.M_branch,
            "hidden_layers": arguments.hidden_layers,
            "hidden_dim_offset": arguments.hidden_dim_offset,
            "train_steps": arguments.train_steps,
            "lr": arguments.lr,
            "seed": seed,
        }
        started = time.perf_counter()
        model_y0, history = train_mdbdp(config)
        elapsed = time.perf_counter() - started
        x0 = torch.zeros(1, arguments.d, device=device)
        t0 = torch.zeros(1, 1, device=device)
        dw0 = torch.zeros(1, arguments.d, device=device)
        with torch.no_grad():
            predicted_y0 = float(model_y0(t0, x0, dw0).item())
        absolute_error = abs(predicted_y0 - true_y0)
        record = RunRecord(
            run_index=run_index,
            seed=seed,
            true_y0=true_y0,
            predicted_y0=predicted_y0,
            absolute_error=absolute_error,
            relative_error=absolute_error / true_y0,
            final_step0_loss=float(history["step_0"]["loss_y"][-1]),
            elapsed_seconds=elapsed,
        )
        records.append(record)
        print(
            f"run={run_index}/{arguments.runs}, seed={seed}, "
            f"Y0={predicted_y0:.8f}, abs-error={absolute_error:.3e}, "
            f"time={elapsed:.1f}s"
        )

    values = np.asarray([record.predicted_y0 for record in records])
    errors = values - true_y0
    summary = SummaryRecord(
        dimension=arguments.d,
        n_steps=arguments.N,
        k_batch=arguments.K_batch,
        m_branch=arguments.M_branch,
        train_steps=arguments.train_steps,
        runs=len(records),
        true_y0=true_y0,
        mean_y0=float(np.mean(values)),
        std_y0=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        relative_error_of_mean=abs(float(np.mean(values)) - true_y0) / true_y0,
        mean_elapsed_seconds=float(
            np.mean([record.elapsed_seconds for record in records])
        ),
    )
    write_csv(records, output_directory / "runs.csv", RunRecord)
    write_csv([summary], output_directory / "summary.csv", SummaryRecord)
    payload = {
        "problem": "Example 1",
        "scheme": "2022 MDBDP complete-tail terminal reconstruction loss",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "runs": [asdict(record) for record in records],
        "summary": asdict(summary),
    }
    with (output_directory / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\nFinal summary")
    print(
        f"trueY0={summary.true_y0:.8f}, meanY0={summary.mean_y0:.8f}, "
        f"std={summary.std_y0:.3e}, MAE={summary.mae:.3e}, "
        f"RMSE={summary.rmse:.3e}, "
        f"RelErr(mean)={summary.relative_error_of_mean:.4%}"
    )
    print(f"outputs={output_directory}")


if __name__ == "__main__":
    main()


