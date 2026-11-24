"""Explicit manual exploratory refits; --validate-only never constructs estimators."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .balancing import balanced_sample_weights, oversample, validate_pool
from .data_loader import ROOT, load_data
from .evaluation import canonical_parameters, test_predictions
from .features import BUDGETS, CLASSES, DESIGN, PREDICTORS, predictor_frame
from .metrics import aggregate_predictions, calculate_metrics
from .models import MODELS, model_module

METHODS = ("class_weighted", "random_oversampling")
SEED = 42
PROTOCOL = "capture_cv_frozen_baseline_imbalance_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_path(root, model, method):
    if model not in MODELS or method not in METHODS:
        raise ValueError("Unknown model or method")
    return Path(root) / "results/ml/imbalance_v2" / model / method


def baseline_parameters(root, model, data):
    """Verify baseline inputs/artifacts and reuse only validation-selected params.

    Explicitly accept CSV CRLF-to-LF matches without rewriting historical hashes.
    Record the exact current hashes and normalized matches in new metadata.
    """
    root = Path(root)
    folder = root / "results/ml" / model
    config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
    marker = folder / "completed.json"
    done = json.loads(marker.read_text(encoding="utf-8"))
    if done.get("status") != "completed" or done.get("model") != model or config.get("model") != model:
        raise ValueError("Invalid completed baseline")
    if not {"configuration.json", "selected_parameters.csv", "validation_search.csv"}.issubset(done["output_sha256"]):
        raise ValueError("Baseline completion omits required files")
    fingerprints, normalized = {}, []
    for name, digest in done["output_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid baseline artifact path")
        path = folder / name
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != digest:
            if path.suffix != ".csv" or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != digest:
                raise ValueError(f"Baseline checksum mismatch: {name}")
            normalized.append(name)
        fingerprints[path.relative_to(root).as_posix()] = actual
    fingerprints[marker.relative_to(root).as_posix()] = sha(marker)
    identity = config["comparison_identity"]
    if (identity["inputs"] != data.hashes or identity["design"] != DESIGN or
            tuple(identity["predictors"]) != PREDICTORS or tuple(identity["labels"]) != CLASSES or
            tuple(identity["budgets"]) != BUDGETS or identity["seed"] != SEED or
            identity["protocol"] != "capture_cv_train_validation_refit_v1"):
        raise ValueError("Baseline differs from approved inputs/protocol")
    selected = pd.read_csv(folder / "selected_parameters.csv")
    search = pd.read_csv(folder / "validation_search.csv")
    keys = {(n, f) for n in BUDGETS for f in (1, 2, 3, 4)}
    if len(selected) != 16 or selected.duplicated(["n_packets", "fold"]).any() or set(zip(selected.n_packets, selected.fold)) != keys:
        raise ValueError("Missing or duplicate baseline selections")
    if set(zip(search.n_packets, search.fold)) != keys:
        raise ValueError("Incomplete validation history")
    grid = {canonical_parameters(p) for p in config["grid"]}
    selections = {}
    for row in selected.itertuples():
        trials = search[(search.n_packets == row.n_packets) & (search.fold == row.fold)].copy()
        trials["parameters"] = trials.parameters.map(lambda s: canonical_parameters(json.loads(s)))
        if len(trials) != len(grid) or set(trials.parameters) != grid or trials.parameters.duplicated().any() or not trials.validation_macro_f1.between(0, 1).all():
            raise ValueError("Invalid baseline validation trials")
        best = trials.sort_values(["validation_macro_f1", "parameters"], ascending=[False, True]).iloc[0]
        params = json.loads(row.parameters)
        if canonical_parameters(params) != best.parameters:
            raise ValueError("Selection disagrees with baseline validation")
        selections[(int(row.n_packets), int(row.fold))] = params
    return selections, fingerprints, normalized


def fitting_pool(train, validation, test):
    """Validate separation before balancing, preprocessing or construction."""
    for left, right in ((train, validation), (train, test), (validation, test)):
        if set(left.flow_id) & set(right.flow_id) or set(left.capture_id) & set(right.capture_id):
            raise ValueError("Flow or capture leakage between partitions")
    fit = pd.concat([train, validation], ignore_index=True).sort_values("flow_id").reset_index(drop=True)
    validate_pool(fit)
    if predictor_frame(fit).isna().all().any():
        raise ValueError("Entirely missing fitting predictor")
    return fit


def validated_partitions(data):
    """Freeze original partitions and reject cross-N drift before any fitting.

    This supplements load_data's manifest checks for direct programmatic callers.
    Only metadata is compared across N; legitimate prefix predictors may differ.
    """
    metadata = ["flow_id", "capture_id", "candidate_class", "verification_status", "label_source"]
    cohort = data.cohort[metadata].reset_index(drop=True).sort_values("flow_id").reset_index(drop=True)
    if cohort.flow_id.duplicated().any():
        raise ValueError("Duplicate cohort flow")
    partitions, reference = {}, {}
    for fold in (1, 2, 3, 4):
        for n in BUDGETS:
            frames = []
            for role in ("train", "validation", "test"):
                frame = data.partition(fold, n, role).reset_index(drop=True).copy(deep=True)
                validate_pool(frame)
                if frame.capture_id.isna().any() or frame.capture_id.astype(str).str.strip().eq("").any():
                    raise ValueError("Missing capture identity")
                identity = frame[metadata].sort_values("flow_id").reset_index(drop=True)
                key = (fold, role)
                if key in reference and not identity.equals(reference[key]):
                    raise ValueError("Flow partition or identity metadata changes across N")
                reference[key] = identity
                partitions[(fold, n, role)] = frame
                frames.append(frame)
            fitting_pool(*frames)
            joined = pd.concat([f[metadata] for f in frames], ignore_index=True).sort_values("flow_id").reset_index(drop=True)
            if not joined.equals(cohort):
                raise ValueError("Partition identities or labels differ from original cohort")
    tests = pd.concat([reference[(fold, "test")] for fold in (1, 2, 3, 4)], ignore_index=True)
    if tests.flow_id.duplicated().any() or not tests.sort_values("flow_id").reset_index(drop=True).equals(cohort):
        raise ValueError("Outer tests must cover each original flow exactly once")
    return partitions


def fit_balanced_pipeline(frame, model, method, params, sampling_seed):
    """Fit only supplied observations; no validation/test argument or access.

    Production calls this only for final train+validation refitting. A future
    independently tuned variant must call it with training only during selection.
    """
    original = frame.reset_index(drop=True).copy(deep=True)
    validate_pool(original)
    sampled = oversample(original.copy(deep=True), sampling_seed) if method == "random_oversampling" else original.copy(deep=True)
    if method not in METHODS or model not in MODELS:
        raise ValueError("Unknown model or method")
    if set(sampled.flow_id) != set(original.flow_id):
        raise ValueError("Resampling omitted originals or introduced foreign observations")
    expected = original.set_index("flow_id").loc[sampled.flow_id].reset_index()
    actual = sampled.reset_index(drop=True).loc[:, expected.columns]
    if not actual.equals(expected):
        raise ValueError("Resampling changed original observation values")
    if method == "random_oversampling":
        counts = sampled.candidate_class.value_counts().reindex(CLASSES)
        if not counts.eq(original.candidate_class.value_counts().max()).all():
            raise ValueError("Resampling failed to balance fitting classes")
    weights = balanced_sample_weights(original) if method == "class_weighted" else None
    pipeline = model_module(model).build(dict(params), seed=SEED)
    options = {}
    if weights is not None:
        if model == "xgboost":
            options["classifier__sample_weight"] = weights
        else:
            pipeline.set_params(classifier__class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(predictor_frame(sampled), sampled.candidate_class.to_numpy(), **options)
    return pipeline, sampled, weights


def run_experiment(data, model, method, selections):
    """Fit-capable; never called during safe validation."""
    output_path(ROOT, model, method)  # Validate names before model dispatch.
    if set(selections) != {(n, f) for n in BUDGETS for f in (1, 2, 3, 4)}:
        raise ValueError("Frozen selections must cover every N/fold")
    selections = {key: dict(params) for key, params in selections.items()}
    partitions = validated_partitions(data)
    pipelines, fit_sets, manifests, roles = {}, {}, [], []
    # Frozen baseline selections are supplied before any fit. Finish all refits
    # before predicting on any outer-test data.
    for n in BUDGETS:
        for fold in (1, 2, 3, 4):
            train, validation, test = (partitions[(fold, n, p)] for p in ("train", "validation", "test"))
            fit = fitting_pool(train, validation, test)
            pipeline, sampled, weights = fit_balanced_pipeline(fit, model, method, selections[(n, fold)], SEED + fold)
            pipelines[(n, fold)] = pipeline
            fit_sets[(n, fold)] = (set(fit.flow_id), set(fit.capture_id))
            manifest = fit[["flow_id", "capture_id", "candidate_class"]].copy()
            manifest["original_partition"] = manifest.flow_id.map({**dict.fromkeys(train.flow_id, "train"), **dict.fromkeys(validation.flow_id, "validation")})
            manifest["multiplicity"] = manifest.flow_id.map(sampled.flow_id.value_counts()).astype(int)
            manifest["sample_weight"] = weights if weights is not None else 1.0
            manifests.append(manifest.assign(n_packets=n, fold=fold, sampling_seed=SEED + fold if method == "random_oversampling" else None))
            counts = train.candidate_class.value_counts().reindex(CLASSES)
            for cls in CLASSES:
                role = "majority" if counts[cls] > counts.median() else "minority" if counts[cls] < counts.median() else "middle"
                roles.append(dict(n_packets=n, fold=fold, candidate_class=cls, training_count=int(counts[cls]), role=role))
    predictions, folds, classes, confusions = [], [], [], []
    for (n, fold), pipeline in pipelines.items():
        test = partitions[(fold, n, "test")]
        pred = test_predictions(pipeline, test, *fit_sets[(n, fold)]).assign(model=model, method=method, n_packets=n, fold=fold)
        expected = test[["flow_id", "capture_id", "candidate_class", "verification_status", "label_source"]].rename(columns={"candidate_class": "y_true"}).reset_index(drop=True)
        if pred.flow_id.duplicated().any() or not pred[list(expected.columns)].equals(expected):
            raise ValueError("Test predictions changed original identities, labels or captures")
        predictions.append(pred)
        summary, per_class, matrix = calculate_metrics(pred.y_true, pred.y_pred)
        folds.append(dict(n_packets=n, fold=fold, **summary))
        classes.append(per_class.assign(n_packets=n, fold=fold))
        for i, actual in enumerate(CLASSES):
            for j, predicted in enumerate(CLASSES):
                confusions.append(dict(n_packets=n, fold=fold, actual_class=actual, predicted_class=predicted,
                                       count=int(matrix[i, j]), row_fraction=float(matrix[i, j] / matrix[i].sum())))
    predictions = pd.concat(predictions, ignore_index=True)
    pooled, pooled_class, pooled_confusion = aggregate_predictions(predictions)
    fold_metrics = pd.DataFrame(folds)
    summary = fold_metrics.groupby("n_packets")[["accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_fold_{stat}" for metric, stat in summary.columns]
    return dict(predictions=predictions, pooled_metrics=pooled, pooled_class_metrics=pooled_class,
                pooled_confusion=pooled_confusion, fold_metrics=fold_metrics, fold_class_metrics=pd.concat(classes, ignore_index=True),
                fold_confusion=pd.DataFrame(confusions), fold_metric_summary=summary.reset_index(),
                training_class_roles=pd.DataFrame(roles), fitting_manifest=pd.concat(manifests, ignore_index=True),
                selected_parameters=pd.DataFrame([dict(n_packets=n, fold=f, parameters=canonical_parameters(p)) for (n, f), p in sorted(selections.items())]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true", help="Read/validate only: no model construction, fitting, predictions or output files")
    return parser.parse_args(argv)


def execute(args):
    root = Path(args.root)
    output = output_path(root, args.model, args.method)
    if output.exists() and not args.validate_only:
        raise FileExistsError(f"Refusing to overwrite previous work: {output}")
    data = load_data(root, expected_flows=851, expected_captures=35)
    selections, baseline_hashes, normalized = baseline_parameters(root, args.model, data)
    validated_partitions(data)
    if args.validate_only:
        print(json.dumps(dict(status="validated_no_training", model=args.model, method=args.method,
                              flows=len(data.cohort), captures=data.cohort.capture_id.nunique(), selections=len(selections),
                              csv_lf_normalized_matches=normalized, output=str(output)), indent=2))
        return
    hashes = {**data.hashes, **baseline_hashes}
    code_hashes = {p.relative_to(root).as_posix(): sha(p) for p in sorted((root / "src/ml").rglob("*.py"))}
    tables = run_experiment(data, args.model, args.method, selections)
    for relative, digest in {**hashes, **code_hashes}.items():
        if sha(root / relative) != digest:
            raise RuntimeError(f"Input or implementation changed during execution: {relative}")
    versions = dict(python=platform.python_version())
    for package in ("numpy", "pandas", "scikit-learn", "xgboost"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    config = dict(protocol=PROTOCOL, model=args.model, method=args.method, design=DESIGN,
                  budgets=BUDGETS, labels=CLASSES, predictors=PREDICTORS, seed=SEED,
                  input_sha256=hashes, implementation_sha256=code_hashes, versions=versions,
                  baseline_csv_lf_normalized_matches=normalized,
                  parameter_selection="Frozen original baseline validation selections; no independent retuning",
                  refit="Fresh complete pipeline on train+validation; all refits before outer-test prediction",
                  weighting="fitting_rows / (4 * fitting_class_count); XGBoost sample_weight, LR/RF balanced class_weight",
                  oversampling="Retain originals; append replacement draws to max count; sorted IDs; seed=42+fold shared across N/models",
                  preprocessing="Imputation and LR scaling fitted on resampled pool for ROS; original fitting pool for weights",
                  isolation="All original partitions frozen and cross-N identities checked before balancing; test predictions must preserve exact metadata",
                  selection_balancing="No new selection; historical unbalanced training-only selection is reused. No weighting/resampling is performed on validation for selection.",
                  limitations="Exploratory after baseline test inspection; unverified capture labels and recording independence")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    done = dict(status="completed", model=args.model, method=args.method,
                output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / "completed.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {args.model}/{args.method}: {output}")


def main(argv=None):
    execute(parse_args(argv))


if __name__ == "__main__":
    main()
