# Corrected imbalance experiments: implementation phase

No research experiments were executed during this change. Historical baseline,
weighted_exploratory and oversampling_exploratory outputs are preserved. The latter
used replacement draws for the entire minority class and could omit originals;
it must not be reported as conventional retain-and-duplicate oversampling.

## Protocol

Use `src.ml.imbalance_experiments` for one model and method per manual invocation.
The existing loader enforces the 851-flow, 35-capture common cohort, four outer
capture-disjoint folds, all four budgets, original labels and seven predictors.
No labels, features, cohort or partitions are regenerated. Recording independence
and capture-derived flow labels remain unverified.

For each N/fold the runner verifies the completed original baseline, its input
fingerprints, and the full validation history. It reuses the original selected
parameters with no new tuning. Candidate baseline fits used training only and
validation macro-F1 selection. These exploratory variants fit a fresh pipeline on
training+validation after those decisions are frozen. All 16 refits finish before
the first outer-test prediction. Test labels or feature values never enter fitting,
weight calculations or sampling. Test identities are used only to reject overlap.

`class_weighted`: LR and RF retain `class_weight="balanced"`. XGBoost receives
`classifier__sample_weight`, aligned with encoded labels, with
`w_i = M / (4 * M_class(i))` computed exclusively from the original fitting pool.
Imputation and LR scaling use that pool; weights affect the classifier only.
XGBoost remains CPU-only (`hist`, `device="cpu"`, `n_jobs=1`).

`random_oversampling`: `src.ml.balancing.oversample` preserves every original row
exactly, appending replacement draws until each class reaches the maximum fitting
class count. IDs are sorted before sampling; seed `42 + fold` is shared across N
and models, preserving multiplicities across budgets. The final classifier seed is
42. The complete pipeline, including imputation and LR scaling, fits the resampled
pool. Weighting and resampling are separate experiments, never combined.

Historical CSV checksums may differ solely because checkout changed LF to CRLF.
The runner accepts a CSV only if either its exact bytes or its explicit CRLF-to-LF
normalization matches the original completion hash. JSON requires an exact match.
No historical hash is rewritten. New metadata records accepted normalized matches
and current exact byte hashes. Substantive artifact changes fail validation.

## Outputs after future approved execution

Each experiment writes exclusively to
`results/ml/imbalance_v2/<model>/<method>/` and refuses an existing directory before
loading or fitting. No historical output paths are used. Incomplete new directories
remain visibly incomplete; they are never automatically deleted or overwritten.

Each directory contains eleven CSVs: `predictions`, `selected_parameters`,
`fitting_manifest`, `training_class_roles`, `pooled_metrics`, `pooled_class_metrics`,
`pooled_confusion`, `fold_metrics`, `fold_class_metrics`, `fold_confusion`, and
`fold_metric_summary`. Predictions retain flow/capture identity and label provenance.
The fitting manifest records every original fitting ID, original train/validation
partition, multiplicity, weight, N, fold and sampling seed. Roles derive from the
original training partition, not balanced/refitted frequencies.

`configuration.json` records frozen-selection policy, protocol, versions, seeds,
data/baseline/code hashes, preprocessing and limitations. Inputs/code are checked
again after execution. `completed.json` is written last with output checksums.
No fitted model serialization or new plots are produced. Existing comparison
scripts do not automatically discover the versioned directories.

## Safe validation and tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_imbalance_experiments.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_ml_framework.py -v
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model xgboost --method class_weighted --validate-only
```

`--validate-only` checks real manifests and baseline artifacts without constructing
estimators, fitting, scoring or writing files. It may be used for any combination.
Tests use synthetic data and mocked classifiers; export tests write temporary
fixtures outside the project. No package installation is automatic.

## Commands for later approval, one experiment at a time

These commands train research models. Do not run during the implementation phase.

```powershell
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model logistic_regression --method random_oversampling
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model random_forest --method random_oversampling
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model xgboost --method class_weighted
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model xgboost --method random_oversampling
```

For provenance-complete LR/RF weighting reruns in separate versioned directories:

```powershell
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model logistic_regression --method class_weighted
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model random_forest --method class_weighted
```

The old `src.ml.oversampling_experiments` command is retired and fails with a
message directing users to this runner; its public `oversample` API uses the shared
corrected function. The historical weighted runner remains unchanged.

These are exploratory fixed-parameter interventions after baseline test inspection,
not independently tuned comparisons or confirmatory evidence. No new performance
claims follow until approved execution and saved-result validation.

## Mandatory isolation checks

Before any estimator construction, the runner freezes all original partitions and
checks pairwise separation by flow and capture, exact original cohort identities
and labels, and unchanged partition assignments across all four N. Each original
flow must appear in outer testing exactly once per N. The existing loader also
cross-checks capture, flow and observation manifests and known recording links.

No new model selection occurs in these variants: historical baseline candidates
fit preprocessing and classifiers on training only and score the original validation
rows. No balancing is retroactively applied to that historical selection. Frozen
parameters come from validation scores, never from outer-test scores. For any
future independently tuned balancing variant, training-only balancing and fresh
training-only preprocessing are mandatory; original validation rows remain untouched.

After selection is frozen, the final fitting pool is train+validation only. Weights,
sampling and preprocessing are recalculated from that pool. The sampler's output
is checked for exact preservation of original row values and IDs, no foreign IDs,
and correct balancing before fitting. Test prediction metadata must exactly match
the frozen original rows, with no duplicate test IDs. The runner never resamples
validation or test prediction inputs and never scores test during selection.

Leakage prevention does not eliminate overfitting. Duplicated training flows can
be memorized and do not add independent information. Capture-disjointness does
not prove recording independence, eliminate label noise, or prevent minority-class
memorization. Later comparisons remain exploratory because baseline test results
have already been inspected.

Additional synthetic adversarial tests:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_imbalance_leakage.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_capture_splits.py -v
```
