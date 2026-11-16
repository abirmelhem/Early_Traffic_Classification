"""Manual execution only: select one model, or validate inputs without any fit."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data_loader import ROOT, load_data
from .evaluation import canonical_parameters, refit_selected, select_configuration, test_predictions
from .features import BUDGETS, CLASSES, DESIGN, PREDICTORS
from .metrics import aggregate_predictions, calculate_metrics
from .models import MODELS, model_module

PROTOCOL_VERSION = "capture_cv_train_validation_refit_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--n", type=int, choices=BUDGETS, help="Run one N; default runs all four")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Read/validate manifests; no model construction, fitting, scoring or outputs")
    return parser.parse_args(argv)


def run_model(data, name, budgets, seed=42):
    """Fit-capable function. Never called by validate-only or by this task's checks."""
    module = model_module(name)
    grid = module.parameter_grid()
    selections, history, events = {}, [], []
    # Complete all validation-based decisions before any outer test prediction.
    for n in budgets:
        for fold in (1, 2, 3, 4):
            train = data.partition(fold, n, "train")
            validation = data.partition(fold, n, "validation")
            params, trials = select_configuration(train, validation, module.build, grid, seed)
            selections[(n, fold)] = params
            history.extend(dict(n_packets=n, fold=fold, **trial) for trial in trials)
            events.append(dict(n_packets=n, fold=fold, phase="selection_training_only",
                               fits=len(grid), flow_ids=";".join(train.flow_id), captures=train.capture_id.nunique()))
    pipelines = {}
    for (n, fold), params in selections.items():
        train = data.partition(fold, n, "train")
        validation = data.partition(fold, n, "validation")
        pipelines[(n, fold)] = refit_selected(train, validation, module.build, params, seed)
        combined = pd.concat([train, validation])
        events.append(dict(n_packets=n, fold=fold, phase="final_training_plus_validation_refit",
                           fits=1, flow_ids=";".join(combined.flow_id), captures=combined.capture_id.nunique()))
    predictions, fold_summaries, class_metrics, confusion, selected, roles = [], [], [], [], [], []
    for (n, fold), pipeline in pipelines.items():
        train = data.partition(fold, n, "train")
        validation = data.partition(fold, n, "validation")
        test = data.partition(fold, n, "test")
        fit = pd.concat([train, validation])
        pred = test_predictions(pipeline, test, fit.flow_id, fit.capture_id).assign(n_packets=n, fold=fold)
        predictions.append(pred)
        summary, per_class, matrix = calculate_metrics(pred.y_true, pred.y_pred)
        fold_summaries.append(dict(n_packets=n, fold=fold, **summary))
        class_metrics.append(per_class.assign(n_packets=n, fold=fold))
        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                confusion.append(dict(n_packets=n, fold=fold, actual_class=actual, predicted_class=predicted,
                                      count=int(matrix[i, j]), row_fraction=float(matrix[i, j]/matrix[i].sum())))
        selected.append(dict(n_packets=n, fold=fold, parameters=canonical_parameters(selections[(n, fold)])))
        counts = train.candidate_class.value_counts().reindex(CLASSES)
        median = counts.median()
        for cls in CLASSES:
            roles.append(dict(n_packets=n, fold=fold, candidate_class=cls, training_count=int(counts[cls]),
                              role="majority" if counts[cls] > median else "minority" if counts[cls] < median else "middle"))
    predictions = pd.concat(predictions, ignore_index=True)
    pooled, pooled_class, pooled_confusion = aggregate_predictions(predictions)
    fold_metrics = pd.DataFrame(fold_summaries)
    means = fold_metrics.groupby("n_packets")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    means.columns = [f"{a}_fold_{b}" for a, b in means.columns]
    return dict(predictions=predictions, fold_metrics=fold_metrics,
                fold_class_metrics=pd.concat(class_metrics, ignore_index=True), fold_confusion=pd.DataFrame(confusion),
                pooled_metrics=pooled, pooled_class_metrics=pooled_class, pooled_confusion=pooled_confusion,
                fold_metric_summary=means.reset_index(), selected_parameters=pd.DataFrame(selected),
                validation_search=pd.DataFrame(history), fit_events=pd.DataFrame(events),
                training_class_roles=pd.DataFrame(roles))


def execute(args):
    data = load_data(args.root, expected_flows=851, expected_captures=35)
    if args.validate_only:
        print(json.dumps(dict(status="validated_no_training", flows=len(data.cohort),
                              captures=data.cohort.capture_id.nunique(), design=DESIGN,
                              predictors=PREDICTORS, label_order=CLASSES, input_sha256=data.hashes), indent=2))
        return
    output = args.root / "results/ml" / args.model
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite previous work: {output}. Archive it manually before a new run.")
    module = model_module(args.model)
    module.build(module.parameter_grid()[0], args.seed)  # Dependency/construction check, no fit.
    budgets = [args.n] if args.n else list(BUDGETS)
    fingerprint = dict(protocol=PROTOCOL_VERSION, inputs=data.hashes, predictors=PREDICTORS,
                       labels=CLASSES, design=DESIGN, budgets=budgets, seed=args.seed)
    tables = run_model(data, args.model, budgets, args.seed)
    # Protect against sources changing while a manually launched experiment runs.
    for relative, digest in data.hashes.items():
        if hashlib.sha256((args.root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Research inputs changed during execution; results not saved")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    versions = {"python": platform.python_version()}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost"]:
        try: versions[package] = version(package)
        except PackageNotFoundError: versions[package] = "not_installed"
    code = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).with_name("models").glob("*.py"))
    config = dict(model=args.model, comparison_identity=fingerprint, versions=versions,
                  grid=module.parameter_grid(), implementation_sha256={p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
                  final_refit="fresh imputation/scaling/classifier on train+validation after selection",
                  scientific_warning="Unverified capture-derived labels and recording independence; no recording-generalization claim")
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    # Written last: incomplete directories cannot be mistaken for finished experiments.
    completion = dict(status="completed", model=args.model,
                      output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output / "completed.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.model}: {output}")


def main(argv=None):
    execute(parse_args(argv))


if __name__ == "__main__":
    main()
