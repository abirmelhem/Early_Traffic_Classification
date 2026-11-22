
"""Exploratory balanced-class-weight comparison with saved baselines."""

import json
from pathlib import Path

import pandas as pd

from .data_loader import ROOT, load_data
from .evaluation import test_predictions
from .features import BUDGETS, predictor_frame
from .metrics import aggregate_predictions
from .models import model_module

MODELS = ("logistic_regression", "random_forest")
OUTPUT = ROOT / "results/ml/weighted_exploratory"


def run():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite previous experiments: {OUTPUT}"
        )

    data = load_data(
        ROOT, expected_flows=851, expected_captures=35
    )

    predictions = []

    for name in MODELS:
        model = model_module(name)

        # Reuse configurations selected in original validation runs.
        configs = pd.read_csv(
            ROOT / "results/ml" / name / "selected_parameters.csv"
        )

        if len(configs) != len(BUDGETS) * 4:
            raise ValueError("Unexpected saved model configurations")

        for n in BUDGETS:
            for fold in (1, 2, 3, 4):

                selected = configs[
                    (configs.n_packets == n)
                    & (configs.fold == fold)
                ]

                if len(selected) != 1:
                    raise ValueError("Missing or duplicate configuration")

                params = json.loads(selected.iloc[0].parameters)

                train = data.partition(fold, n, "train")
                validation = data.partition(fold, n, "validation")
                test = data.partition(fold, n, "test")

                # Fit on training + validation, never on test.
                fit = pd.concat(
                    [train, validation], ignore_index=True
                )

                if set(fit.capture_id) & set(test.capture_id):
                    raise ValueError("Capture leakage detected")

                pipeline = model.build(params, seed=42)

                # Main difference from the original baseline.
                pipeline.set_params(
                    classifier__class_weight="balanced"
                )

                pipeline.fit(
                    predictor_frame(fit),
                    fit.candidate_class
                )

                pred = test_predictions(
                    pipeline, test, fit.flow_id, fit.capture_id
                )

                pred["model"] = name
                pred["n_packets"] = n
                pred["fold"] = fold

                predictions.append(pred)

                print(
                    f"Completed {name}: N={n}, fold={fold}",
                    flush=True
                )

    predictions = pd.concat(predictions, ignore_index=True)

    summary_tables = []
    class_tables = []

    for name in MODELS:
        subset = predictions[predictions.model == name]

        summary, per_class, _ = aggregate_predictions(subset)

        summary_tables.append(
            summary.assign(
                model=name, method="balanced_class_weights"
            )
        )

        class_tables.append(
            per_class.assign(
                model=name, method="balanced_class_weights"
            )
        )

    weighted = pd.concat(summary_tables, ignore_index=True)
    by_class = pd.concat(class_tables, ignore_index=True)

    comparisons = []

    for name in MODELS:
        baseline = pd.read_csv(
            ROOT / "results/ml" / name / "pooled_metrics.csv"
        )

        old = baseline[
            ["n_packets", "macro_f1"]
        ].rename(columns={
            "macro_f1": "baseline_macro_f1"
        })

        new = weighted[
            weighted.model == name
        ][["n_packets", "macro_f1"]].rename(columns={
            "macro_f1": "weighted_macro_f1"
        })

        merged = old.merge(
            new, on="n_packets", validate="one_to_one"
        )

        merged["model"] = name

        merged["delta_macro_f1"] = (
            merged.weighted_macro_f1
            - merged.baseline_macro_f1
        )

        comparisons.append(merged)

    comparison = pd.concat(comparisons, ignore_index=True)

    OUTPUT.mkdir(parents=True, exist_ok=False)

    predictions.to_csv(
        OUTPUT / "predictions.csv", index=False
    )

    weighted.to_csv(
        OUTPUT / "weighted_metrics.csv", index=False
    )

    by_class.to_csv(
        OUTPUT / "weighted_per_class.csv", index=False
    )

    comparison.to_csv(
        OUTPUT / "baseline_comparison.csv", index=False
    )

    (OUTPUT / "README.md").write_text(
        "Exploratory class-weighted experiment. "
        "Original validation-selected hyperparameters are reused. "
        "Class weights are computed from the train+validation "
        "partition, never test data. Baseline test results were "
        "already inspected, so new comparisons are exploratory. "
        "Capture-derived labels remain unverified. "
        "Original baseline results are unchanged.\n",
        encoding="utf-8",
    )

    print("\nWeighted experiments completed!")
    print(comparison.round(3).to_string(index=False))
    print(f"\nResults saved in: {OUTPUT}")


if __name__ == "__main__":
    run()
