import requests
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import argparse

# Constants
API_URL = "http://hyperturing.stanford.edu:8000"
CACHE_FILE = "scaling_data.json"
API_KEY_PATH = os.path.expanduser("~/.ssh/id_rsa.pub")


def get_api_key():
    try:
        with open(API_KEY_PATH, "r") as f:
            return f.read().strip()
    except Exception:
        return None


class ExperimentRunner:
    def __init__(self, api_key: str, cache_file: str = CACHE_FILE, mock: bool = False):
        self.api_key = api_key
        self.cache_file = cache_file
        self.mock = mock
        self.data = self._load_cache()
        self.flops_used = 0  # Track session usage, total usage is from API

    def _load_cache(self) -> List[Dict]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def _get_cached_result(self, config: Dict) -> Optional[float]:
        for run in self.data:
            # Check if config matches (excluding loss/flops info)
            match = True
            for k, v in config.items():
                if k not in ["loss", "total_flops_used"] and run.get(k) != v:
                    match = False
                    break
            if match:
                return run["loss"]
        return None

    def query_loss(self, config: Dict) -> float:
        # Check cache first
        cached_loss = self._get_cached_result(config)
        if cached_loss is not None:
            print(f"Cache hit for config: {config}")
            return cached_loss

        if self.mock:
            print(f"Mocking query for: {config}")
            # Mock loss function: roughly Chinchilla scaling
            N = 12 * config["num_layers"] * config["d_model"] ** 2
            C = config["train_flops"]
            D = C / (6 * N)
            # A simple power law proxy
            loss = 1.0 + (N / 1e9) ** -0.3 + (D / 1e9) ** -0.3
            time.sleep(0.1)
            result = {"loss": loss, "total_flops_used": self.flops_used + C}
        else:
            print(f"Querying API for: {config}")
            params = config.copy()
            params["api_key"] = self.api_key
            try:
                response = requests.get(f"{API_URL}/loss", params=params, timeout=10)
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    return float("inf")
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                return float("inf")

        loss = result["loss"]

        # Save to cache
        record = config.copy()
        record["loss"] = loss
        self.data.append(record)
        self._save_cache()

        return loss

    def get_total_flops(self):
        if self.mock:
            return sum(r["train_flops"] for r in self.data)

        try:
            response = requests.get(
                f"{API_URL}/total_flops_used",
                params={"api_key": self.api_key},
                timeout=5,
            )
            if response.status_code == 200:
                return float(response.json())
        except:
            pass
        return 0.0


def generate_search_space():
    # Budgets to test
    # We want to fit N_opt ~ C^a
    # Predicted N_opt from Chinchilla: N ~ C^0.46
    # C=1e15 -> N ~ 13M
    # C=1e16 -> N ~ 38M
    # C=3e16 -> N ~ 63M
    # C=1e17 -> N ~ 110M

    configs = []

    # 1. Budget 1e15
    flops = 1e15
    # Try N around 5M - 25M
    # d_model: 128, layers: 4 -> N ~ 0.8M (too small)
    # d_model: 256, layers: 2 -> N ~ 1.5M
    # d_model: 256, layers: 4 -> N ~ 3M
    # d_model: 256, layers: 8 -> N ~ 6M
    # d_model: 512, layers: 4 -> N ~ 12M
    # d_model: 512, layers: 8 -> N ~ 25M

    for d_model in [256, 512]:
        for layers in [4, 6, 8, 12]:
            if d_model == 256 and layers > 8:
                continue
            configs.append(
                {
                    "d_model": d_model,
                    "num_layers": layers,
                    "num_heads": 8,
                    "batch_size": 128,
                    "train_flops": int(flops),
                }
            )

    # 2. Budget 3e15
    flops = 3e15
    for d_model in [256, 512]:
        for layers in [6, 8, 12, 16]:
            configs.append(
                {
                    "d_model": d_model,
                    "num_layers": layers,
                    "num_heads": 8,
                    "batch_size": 128,
                    "train_flops": int(flops),
                }
            )

    # 3. Budget 1e16
    flops = 1e16
    # Target N ~ 40M
    # d_model 512, layers 12 -> 37M
    for d_model in [512, 768]:
        for layers in [8, 12, 16, 20]:
            configs.append(
                {
                    "d_model": d_model,
                    "num_layers": layers,
                    "num_heads": 8,
                    "batch_size": 128,
                    "train_flops": int(flops),
                }
            )

    # Add learning rate variation (simple grid for now, or iterative?)
    # For simplicity, let's just pick a few LRs for each config?
    # Or better: just run one LR (e.g. 5e-4) initially, then fine tune?
    # The prompt says "analyze how hyperparameters affect training loss".
    # Let's add LR dimension.

    final_configs = []
    for c in configs:
        for lr in [1e-3, 5e-4, 1e-4]:
            c_new = c.copy()
            c_new["learning_rate"] = lr
            final_configs.append(c_new)

    # Sort by compute cost (low to high) to fail fast if budget exceeded
    final_configs.sort(key=lambda x: x["train_flops"])

    # Filter to ensure we don't massively exceed budget
    # approx_cost = sum(c['train_flops'] for c in final_configs)
    # print(f"Planned cost: {approx_cost:.2e}")

    return final_configs


def analyze_results(runner):
    data = runner.data
    if not data:
        print("No data to analyze.")
        return

    # Group by budget
    budget_groups = defaultdict(list)
    for run in data:
        if run["loss"] != float("inf"):
            budget_groups[run["train_flops"]].append(run)

    opt_points = []
    print("\nOptimal configurations found:")
    print(f"{'Budget':<10} | {'Params':<15} | {'Loss':<10} | {'Config'}")

    for budget in sorted(budget_groups.keys()):
        runs = budget_groups[budget]
        best_run = min(runs, key=lambda x: x["loss"])
        N = 12 * best_run["num_layers"] * best_run["d_model"] ** 2
        print(f"{budget:<10.0e} | {N:<15.2e} | {best_run['loss']:<10.4f} | {best_run}")
        opt_points.append({"C": budget, "N": N, "loss": best_run["loss"]})

    # Fit scaling law N = A * C^a
    if len(opt_points) > 1:
        C_vals = np.array([p["C"] for p in opt_points])
        N_vals = np.array([p["N"] for p in opt_points])

        coeffs = np.polyfit(np.log(C_vals), np.log(N_vals), 1)
        a = coeffs[0]
        A = np.exp(coeffs[1])

        print(f"\nScaling Law: N_opt = {A:.4e} * C^{a:.4f}")

        # Predict for 1e19
        C_target = 1e19
        N_target = A * (C_target**a)
        print(f"Predicted Optimal N for 1e19: {N_target:.2e}")

        # Suggest hyperparameters for N_target
        # N = 12 * layers * d_model^2
        # Try to find integer layers/d_model close to N_target
        best_diff = float("inf")
        best_config = None

        for d in range(64, 4096, 64):  # Extended range for extrapolation
            # N ~ 12 * L * d^2 => L ~ N / (12 * d^2)
            L_float = N_target / (12 * d**2)
            L = round(L_float)
            if L < 2:
                continue

            # Prefer configurations closer to valid range if possible, but allow larger
            N_est = 12 * L * d**2
            diff = abs(N_est - N_target)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    "d_model": d,
                    "num_layers": L,
                    "num_heads": 16,
                }  # num_heads max for flexibility

        print(f"Suggested Configuration (Hypothetical): {best_config}")

        # Plot
        plt.figure()
        plt.scatter(C_vals, N_vals, label="Experiments")
        C_grid = np.logspace(14, 20, 100)
        plt.plot(C_grid, A * C_grid**a, "r--", label="Fit")
        plt.scatter(
            [C_target],
            [N_target],
            color="green",
            marker="*",
            s=200,
            label="Target (1e19)",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Compute (FLOPs)")
        plt.ylabel("Parameters")
        plt.legend()
        plt.savefig("scaling_law_fit.png")


import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print planned runs"
    )
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key and not args.mock:
        print("Error: valid API key not found in ~/.ssh/id_rsa.pub")
        return

    runner = ExperimentRunner(api_key, mock=args.mock)
    configs = generate_search_space()

    # Filter out already done
    todo_configs = []
    for c in configs:
        if runner._get_cached_result(c) is None:
            todo_configs.append(c)

    print(f"Total configs planned: {len(configs)}")
    print(f"Configs remaining: {len(todo_configs)}")

    if args.dry_run:
        est_cost = sum(c["train_flops"] for c in todo_configs)
        print(f"Estimated cost of remaining runs: {est_cost:.2e} FLOPs")
        return

    # Run experiments
    flops_limit = 2e18
    # current_usage = runner.get_total_flops() # This might be slow/fail if API down
    # Just execute and let API/Loop handle limits roughly

    total_new_flops = 0
    for i, config in enumerate(todo_configs):
        print(f"Running {i+1}/{len(todo_configs)} ...")

        # Safety check locally too
        if total_new_flops + config["train_flops"] > flops_limit:
            print("Local budget limit reached. Stopping.")
            break

        loss = runner.query_loss(config)

        if loss != float("inf"):
            total_new_flops += config["train_flops"]
        else:
            print("Aborting due to API error (or maybe budget exceeded)")
            break

    analyze_results(runner)


if __name__ == "__main__":
    main()
