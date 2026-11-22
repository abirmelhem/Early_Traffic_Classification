
import json

import numpy as np
import pandas as pd

from .data_loader import ROOT, load_data
from .evaluation import test_predictions
from .features import BUDGETS, CLASSES, predictor_frame
from .metrics import aggregate_predictions, calculate_metrics
from .models import model_module

MODELS = ("logistic_regression", "random_forest")
OUTPUT = ROOT / "results/ml/oversampling_exploratory"
SEED = 42


def oversample(frame, seed):
    rng = np.random.default_rng(seed)
    counts = frame.candidate_class.value_counts()
    target = int(counts.max())
    parts = []

    for cls in CLASSES:
        group = frame[frame.candidate_class == cls]

        if group.empty:
            raise ValueError(f"Missing training class: {cls}")

        positions = rng.choice(
            len(group),
            size=target,
            replace=len(group) < target,
        )
        parts.append(group.iloc[positions].copy())

    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Results already exist: {OUTPUT}"
        )

    data = load_data(
        ROOT, expected_flows=851, expected_captures=35
    )

    predictions = []
    fold_metrics = []

    for name in MODELS:
        module = model_module(name)

        configs = pd.read_csv(
            ROOT / "results/ml" / name / "selected_parameters.csv"
        )

        for n in BUDGETS:
            for fold in (1, 2, 3, 4):
                selected = configs[
                    (configs.n_packets == n)
                    & (configs.fold == fold)
                ]

                if len(selected) != 1:
                    raise ValueError("Invalid model configuration")

                params = json.loads(
                    selected.iloc[0].parameters
                )

                train = data.partition(fold, n, "train")
                validation = data.partition(
                    fold, n, "validation"
                )
                test = data.partition(fold, n, "test")

                fit = pd.concat(
                    [train, validation], ignore_index=True
                )

                if set(fit.capture_id) & set(test.capture_id):
                    raise ValueError("Capture leakage detected")

                balanced = oversample(
                    fit, seed=SEED + fold + n
                )

                pipeline = module.build(params, seed=SEED)

                pipeline.fit(
                    predictor_frame(balanced),
                    balanced.candidate_class
                )

                result = test_predictions(
                    pipeline,
                    test,
                    fit.flow_id,
                    fit.capture_id,
                )

                result = result.assign(
                    model=name,
                    n_packets=n,
                    fold=fold,
                )

                predictions.append(result)

                summary, _, _ = calculate_metrics(
                    result.y_true, result.y_pred
                )

                fold_metrics.append({
                    "model": name,
                    "n_packets": n,
                    "fold": fold,
                    **summary,
                })

                print(
                    f"Completed {name}: N={n}, fold={fold}",
                    flush=True,
                )

    predictions = pd.concat(
        predictions, ignore_index=True
    )

    summary_tables = []
    class_tables = []

    for name in MODELS:
        subset = predictions[
            predictions.model == name
        ]

        summary, per_class, _ = aggregate_predictions(
            subset
        )

        summary_tables.append(
            summary.assign(model=name)
        )
        class_tables.append(
            per_class.assign(model=name)
        )

    metrics = pd.concat(
        summary_tables, ignore_index=True
    )
    per_class = pd.concat(
        class_tables, ignore_index=True
    )

    comparisons = []

    for name in MODELS:
        original = pd.read_csv(
            ROOT / "results/ml" / name / "pooled_metrics.csv"
        )

        weighted = pd.read_csv(
            ROOT
            / "results/ml/weighted_exploratory/weighted_metrics.csv"
        )

        baseline = original[
            ["n_packets", "macro_f1"]
        ].rename(columns={
            "macro_f1": "baseline_f1"
        })

        weighted = weighted[
            weighted.model == name
        ][["n_packets", "macro_f1"]].rename(
            columns={"macro_f1": "weighted_f1"}
        )

        ros = metrics[
            metrics.model == name
        ][["n_packets", "macro_f1"]].rename(
            columns={"macro_f1": "oversampled_f1"}
        )

        comparison = (
            baseline
            .merge(weighted, on="n_packets",
                   validate="one_to_one")
            .merge(ros, on="n_packets",
                   validate="one_to_one")
        )

        comparison["model"] = name
        comparison["delta_vs_baseline"] = (
            comparison.oversampled_f1
            - comparison.baseline_f1
        )

        comparisons.append(comparison)

    comparison = pd.concat(
        comparisons, ignore_index=True
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    predictions.to_csv(
        OUTPUT / "predictions.csv", index=False
    )
    metrics.to_csv(
        OUTPUT / "oversampled_metrics.csv", index=False
    )
    per_class.to_csv(
        OUTPUT / "oversampled_per_class.csv", index=False
    )
    pd.DataFrame(fold_metrics).to_csv(
        OUTPUT / "fold_metrics.csv", index=False
    )
    comparison.to_csv(
        OUTPUT / "three_method_comparison.csv",
        index=False,
    )

    print("\nRandom oversampling completed!")
    print(comparison.round(3).to_string(index=False))
    print(f"\nResults saved in: {OUTPUT}")


if __name__ == "__main__":
    run()
