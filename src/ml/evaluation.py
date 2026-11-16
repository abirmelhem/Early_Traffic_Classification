"""Selection, approved train+validation refit, and isolated test scoring stages."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .features import CLASSES, predictor_frame
from .metrics import calculate_metrics


def canonical_parameters(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fit_pipeline(builder, params, seed, frame):
    X = predictor_frame(frame)
    if X.isna().all().any():
        raise ValueError("Training predictor is entirely missing; investigate before fitting")
    if set(frame.candidate_class) != set(CLASSES):
        raise ValueError("Training requires all four classes")
    pipeline = builder(dict(params), seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, frame.candidate_class.to_numpy())
    return pipeline


def select_configuration(train, validation, builder, grid, seed):
    """No test argument: each candidate pipeline fits training rows only."""
    if set(train.flow_id) & set(validation.flow_id) or set(train.capture_id) & set(validation.capture_id):
        raise ValueError("Selection train/validation overlap")
    if not grid:
        raise ValueError("Empty hyperparameter grid")
    rows = []
    for params in sorted(grid, key=canonical_parameters):
        pipeline = fit_pipeline(builder, params, seed, train)
        pred = pipeline.predict(predictor_frame(validation))
        score = calculate_metrics(validation.candidate_class, pred)[0]["macro_f1"]
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation score")
        rows.append(dict(parameters=canonical_parameters(params), validation_macro_f1=score))
    best = min(rows, key=lambda r: (-r["validation_macro_f1"], r["parameters"]))
    return json.loads(best["parameters"]), rows


def refit_selected(train, validation, builder, params, seed):
    """Explicit user-requested protocol: fresh preprocessing and model on train+val."""
    if set(train.flow_id) & set(validation.flow_id) or set(train.capture_id) & set(validation.capture_id):
        raise ValueError("Refit train/validation overlap")
    combined = pd.concat([train, validation], axis=0)
    return fit_pipeline(builder, params, seed, combined)


def test_predictions(pipeline, test, fit_ids, fit_captures):
    if set(test.flow_id) & set(fit_ids) or set(test.capture_id) & set(fit_captures):
        raise ValueError("Outer test overlaps final refit data")
    pred = pipeline.predict(predictor_frame(test))  # Exactly one test prediction call.
    output = test[["flow_id", "capture_id", "candidate_class", "verification_status", "label_source"]].copy()
    output = output.rename(columns={"candidate_class": "y_true"})
    output["y_pred"] = pred
    return output.reset_index(drop=True)
