
"""Compare classification performance with packet observation delay."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DELAY_FILE = ROOT / "results/supervised_delay/supervised_delay_overall.csv"
METRICS_FILE = ROOT / "results/ml/comparisons/pooled_model_comparison.csv"
OUTPUT = ROOT / "results/reliability_delay"

BUDGETS = (3, 5, 10, 20)
MODELS = ("logistic_regression", "random_forest", "xgboost")


def run():
    delay = pd.read_csv(DELAY_FILE)
    metrics = pd.read_csv(METRICS_FILE)

    # Check that required columns exist.
    for label, frame, columns in (
        ("delay", delay, ["n_packets", "flows", "median_ms",
                          "mean_ms", "p90_ms", "p95_ms"]),
        ("metrics", metrics, ["n_packets", "model",
                              "macro_f1", "support"]),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Missing {label} columns: {sorted(missing)}"
            )

    # Validate the supervised-cohort delay measurements.
    if delay.n_packets.duplicated().any():
        raise ValueError("Duplicate packet budgets in delay data")

    if set(delay.n_packets) != set(BUDGETS):
        raise ValueError("Expected packet budgets 3, 5, 10, 20")

    if not (delay.flows == 851).all():
        raise ValueError("Expected 851 supervised flows")

    if (delay.median_ms < 0).any():
        raise ValueError("Invalid negative observation delay")

    if (delay.p95_ms < delay.median_ms).any():
        raise ValueError("Invalid delay percentiles")

    # Select and validate the existing model results.
    subset = metrics[metrics.model.isin(MODELS)].copy()

    expected = {
        (model, n)
        for model in MODELS
        for n in BUDGETS
    }

    actual = set(zip(subset.model, subset.n_packets))

    if actual != expected:
        raise ValueError("Missing model or packet-budget results")

    if subset.duplicated(["model", "n_packets"]).any():
        raise ValueError("Duplicate model results")

    if not (subset.support == 851).all():
        raise ValueError("Model results use a different cohort")

    if not subset.macro_f1.between(0, 1).all():
        raise ValueError("Invalid macro-F1 values")

    # Combine classification results and observation delay.
    comparison = subset.merge(
        delay[[
            "n_packets", "median_ms",
            "mean_ms", "p90_ms", "p95_ms"
        ]],
        on="n_packets",
        how="left",
        validate="many_to_one",
    ).sort_values(["model", "n_packets"])

    # Compare every packet budget against N=5.
    baseline = comparison[
        comparison.n_packets == 5
    ][["model", "macro_f1", "median_ms"]].rename(
        columns={
            "macro_f1": "f1_n5",
            "median_ms": "median_n5_ms",
        }
    )

    comparison = comparison.merge(
        baseline,
        on="model",
        validate="many_to_one",
    )

    comparison["delta_f1_vs_n5"] = (
        comparison.macro_f1 - comparison.f1_n5
    )

    comparison["extra_median_delay_vs_n5_ms"] = (
        comparison.median_ms - comparison.median_n5_ms
    )

    comparison = comparison.drop(
        columns=["f1_n5", "median_n5_ms"]
    )

    # Save the results.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUT / "reliability_delay_comparison.csv",
        index=False,
    )

    # Generate comparison figures.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for model in MODELS:
        data = comparison[
            comparison.model == model
        ].sort_values("n_packets")

        label = model.replace("_", " ").title()

        axes[0].plot(
            data.n_packets,
            data.macro_f1,
            marker="o",
            label=label,
        )

        axes[1].plot(
            data.median_ms,
            data.macro_f1,
            marker="o",
            label=label,
        )

        for row in data.itertuples():
            axes[1].annotate(
                f"N={int(row.n_packets)}",
                (row.median_ms, row.macro_f1),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    axes[0].set(
        xlabel="Observed packets (N)",
        ylabel="Pooled outer-test macro-F1",
        title="Reliability by Packet Budget",
    )
    axes[0].set_xticks(BUDGETS)

    axes[1].set(
        xlabel="Median observation delay (ms)",
        ylabel="Pooled outer-test macro-F1",
        title="Reliability vs. Observation Delay",
    )
    axes[1].set_xscale("log")

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "reliability_delay_comparison.png",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)

    print("Reliability-versus-delay analysis completed!")
    print(
        comparison[[
            "model", "n_packets", "macro_f1",
            "median_ms", "delta_f1_vs_n5"
        ]].round(3).to_string(index=False)
    )
    print(f"Results saved in: {OUTPUT}")


if __name__ == "__main__":
    run()
