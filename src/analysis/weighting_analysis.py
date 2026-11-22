
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/ml/weighting_analysis"
MODELS = ["logistic_regression", "random_forest"]
BUDGETS = [3, 5, 10, 20]
ROLES = {
    "CHAT": "minority",
    "FT": "minority",
    "MAIL": "majority",
    "VOIP": "majority",
}

baseline = pd.read_csv(
    ROOT / "results/ml/comparisons/research_analysis/tables/pooled_per_class.csv"
)
weighted = pd.read_csv(
    ROOT / "results/ml/weighted_exploratory/weighted_per_class.csv"
)

keys = ["model", "n_packets", "candidate_class"]
baseline = baseline[baseline.model.isin(MODELS)]
weighted = weighted[weighted.model.isin(MODELS)]

comparison = baseline[keys + ["f1", "support"]].merge(
    weighted[keys + ["f1", "support"]],
    on=keys,
    suffixes=("_baseline", "_weighted"),
    validate="one_to_one",
)

assert len(comparison) == 32
assert (comparison.support_baseline == comparison.support_weighted).all()
assert set(comparison.n_packets) == set(BUDGETS)

comparison["role"] = comparison.candidate_class.map(ROLES)
comparison["delta_f1"] = (
    comparison.f1_weighted - comparison.f1_baseline
)

role_summary = (
    comparison.groupby(["model", "n_packets", "role"])
    .agg(
        baseline_f1=("f1_baseline", "mean"),
        weighted_f1=("f1_weighted", "mean"),
    )
    .reset_index()
)

role_summary["delta_f1"] = (
    role_summary.weighted_f1 - role_summary.baseline_f1
)

OUTPUT.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT / "per_class_comparison.csv", index=False
)
role_summary.to_csv(
    OUTPUT / "majority_minority_comparison.csv", index=False
)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

for i, model in enumerate(MODELS):
    for j, role in enumerate(["minority", "majority"]):
        ax = axes[i, j]
        data = role_summary[
            (role_summary.model == model)
            & (role_summary.role == role)
        ].sort_values("n_packets")

        ax.plot(
            data.n_packets, data.baseline_f1,
            marker="o", label="Original"
        )
        ax.plot(
            data.n_packets, data.weighted_f1,
            marker="s", label="Class-weighted"
        )

        ax.set_title(f"{model.replace('_', ' ').title()} — {role.title()}")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Observed packets (N)")
        ax.set_ylabel("Mean pooled class F1")
        ax.grid(alpha=0.2)
        ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT / "weighting_comparison.png",
    dpi=300,
    facecolor="white",
)
plt.close(fig)

print("\nClass-weighting analysis completed!")
print("\nMajority/minority comparison:")
print(role_summary.round(3).to_string(index=False))
print("\nPer-class comparison:")
print(
    comparison[
        keys + ["f1_baseline", "f1_weighted", "delta_f1"]
    ].round(3).to_string(index=False)
)
print(f"\nResults saved in: {OUTPUT}")
