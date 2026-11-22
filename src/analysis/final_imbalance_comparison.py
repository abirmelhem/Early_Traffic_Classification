
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/ml/final_imbalance_comparison"

MODELS = ["logistic_regression", "random_forest"]
BUDGETS = [3, 5, 10, 20]
CLASSES = ["CHAT", "MAIL", "VOIP", "FT"]
METHODS = ["Original", "Class Weighting", "Random Oversampling"]

paths = {
    "Original": ROOT / "results/ml/comparisons/research_analysis/tables/pooled_per_class.csv",
    "Class Weighting": ROOT / "results/ml/weighted_exploratory/weighted_per_class.csv",
    "Random Oversampling": ROOT / "results/ml/oversampling_exploratory/oversampled_per_class.csv",
}

tables = []
keys = ["model", "n_packets", "candidate_class"]

for method, path in paths.items():
    df = pd.read_csv(path)
    df = df[df.model.isin(MODELS)].copy()

    if len(df) != 32 or df.duplicated(keys).any():
        raise ValueError(f"Invalid results for {method}")

    expected = {
        (m, n, c)
        for m in MODELS for n in BUDGETS for c in CLASSES
    }

    if set(map(tuple, df[keys].to_numpy())) != expected:
        raise ValueError(f"Missing combinations for {method}")

    df["method"] = method
    tables.append(df[keys + ["method", "f1", "support"]])

data = pd.concat(tables, ignore_index=True)

if data["f1"].isna().any() or not data["f1"].between(0, 1).all():
    raise ValueError("Invalid F1 scores")

supports = data.groupby(keys)["support"].nunique()
if not (supports == 1).all():
    raise ValueError("Inconsistent class support")

data["role"] = data.candidate_class.map({
    "CHAT": "Minority",
    "FT": "Minority",
    "MAIL": "Majority",
    "VOIP": "Majority",
})

role_summary = (
    data.groupby(
        ["model", "n_packets", "method", "role"]
    )["f1"]
    .mean()
    .reset_index(name="mean_pooled_class_f1")
)

macro_summary = (
    data.groupby(
        ["model", "n_packets", "method"]
    )["f1"]
    .mean()
    .reset_index(name="macro_f1")
)

comparison = macro_summary.pivot(
    index=["model", "n_packets"],
    columns="method",
    values="macro_f1",
).reset_index()

comparison.columns.name = None

comparison["weighting_change"] = (
    comparison["Class Weighting"] - comparison["Original"]
)
comparison["oversampling_change"] = (
    comparison["Random Oversampling"] - comparison["Original"]
)

OUTPUT.mkdir(parents=True, exist_ok=True)

data.to_csv(OUTPUT / "all_per_class_results.csv", index=False)
role_summary.to_csv(
    OUTPUT / "majority_minority_results.csv", index=False
)
comparison.to_csv(
    OUTPUT / "three_method_macro_comparison.csv", index=False
)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for ax, model in zip(axes, MODELS):
    subset = macro_summary[macro_summary.model == model]

    for method in METHODS:
        s = subset[
            subset.method == method
        ].sort_values("n_packets")

        ax.plot(
            s.n_packets,
            s.macro_f1,
            marker="o",
            label=method,
        )

    ax.set_title(model.replace("_", " ").title())
    ax.set_ylabel("Pooled macro-F1")
    ax.set_ylim(0, 1)
    ax.set_xticks(BUDGETS)
    ax.grid(alpha=0.2)
    ax.legend()

axes[-1].set_xlabel("Observed packets (N)")
fig.tight_layout()
fig.savefig(
    OUTPUT / "three_method_macro_f1.png",
    dpi=300,
    facecolor="white",
)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

for i, model in enumerate(MODELS):
    for j, role in enumerate(["Minority", "Majority"]):
        ax = axes[i, j]

        subset = role_summary[
            (role_summary.model == model)
            & (role_summary.role == role)
        ]

        for method in METHODS:
            s = subset[
                subset.method == method
            ].sort_values("n_packets")

            ax.plot(
                s.n_packets,
                s.mean_pooled_class_f1,
                marker="o",
                label=method,
            )

        ax.set_title(
            f"{model.replace('_', ' ').title()} — {role}"
        )
        ax.set_xticks(BUDGETS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Observed packets (N)")
        ax.set_ylabel("Mean pooled per-class F1")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(
    OUTPUT / "three_method_majority_minority.png",
    dpi=300,
    facecolor="white",
)
plt.close(fig)

print("\nFinal imbalance comparison completed!")
print(comparison.round(3).to_string(index=False))
print(f"\nResults saved in: {OUTPUT}")
