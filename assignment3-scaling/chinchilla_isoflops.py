import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os


def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def analyze_isoflops(data):
    # Group by compute budget
    budget_groups = defaultdict(list)
    for run in data:
        budget_groups[run["compute_budget"]].append(run)

    # Find optimal model size for each budget
    opt_data = []

    sorted_budgets = sorted(budget_groups.keys())

    print(
        f"{'Budget (FLOPs)':<20} | {'Opt Params (N)':<20} | {'Opt Tokens (D)':<20} | {'Min Loss':<10}"
    )
    print("-" * 80)

    for budget in sorted_budgets:
        runs = budget_groups[budget]
        # Find run with minimum loss
        best_run = min(runs, key=lambda x: x["final_loss"])

        N_opt = best_run["parameters"]
        loss = best_run["final_loss"]

        # Calculate D_opt using C = 6ND => D = C / (6N)
        D_opt = budget / (6 * N_opt)

        opt_data.append({"C": budget, "N_opt": N_opt, "D_opt": D_opt, "loss": loss})

        print(f"{budget:<20.2e} | {N_opt:<20.2e} | {D_opt:<20.2e} | {loss:<10.4f}")

    return opt_data


def fit_power_law(x, y):
    # Fit y = A * x^b
    # log(y) = log(A) + b * log(x)
    log_x = np.log(x)
    log_y = np.log(y)

    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    log_A = coeffs[1]
    A = np.exp(log_A)

    return A, b


def plot_scaling_law(opt_data, target_variable, target_label, filename):
    C_values = np.array([d["C"] for d in opt_data])
    target_values = np.array([d[target_variable] for d in opt_data])

    # Fit power law
    A, b = fit_power_law(C_values, target_values)
    print(f"\nScaling Law for {target_label}: {target_variable} = {A:.4e} * C^{b:.4f}")

    # Plot data points
    plt.figure(figsize=(10, 6))
    plt.scatter(C_values, target_values, color="blue", label="Experimental Optima")

    # Plot fit line (extrapolated)
    C_extrap = np.logspace(np.log10(min(C_values)), 24, 100)  # Extrapolate to 1e24
    target_pred = A * (C_extrap**b)

    plt.plot(
        C_extrap,
        target_pred,
        color="red",
        linestyle="--",
        label=rf"Fit: ${A:.2e} \cdot C^{{{b:.2f}}}$",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Compute Budget (FLOPs)")
    plt.ylabel(f"Optimal {target_label}")
    plt.title(f"Scaling Law: Optimal {target_label} vs Compute")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    plt.savefig(filename)
    print(f"Saved plot to {filename}")

    # Predict for specific budgets
    for budget in [1e23, 1e24]:
        pred = A * (budget**b)
        print(f"Predicted Optimal {target_label} for C={budget:.0e}: {pred:.4e}")


def main():
    data_path = "data/isoflops_curves.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = load_data(data_path)
    opt_data = analyze_isoflops(data)

    print("\n--- Model Size Scaling ---")
    plot_scaling_law(
        opt_data, "N_opt", "Model Size (Parameters)", "isoflops_model_size.png"
    )

    print("\n--- Dataset Size Scaling ---")
    plot_scaling_law(
        opt_data, "D_opt", "Dataset Size (Tokens)", "isoflops_dataset_size.png"
    )


if __name__ == "__main__":
    main()
