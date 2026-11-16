# Modular first-N ML framework — implemented, research experiments not executed

## Protocol revision authorized by the current implementation request

Earlier readiness/split documents specify no train+validation refit. The current
request explicitly supersedes that choice: candidates fit training only; validation
macro-F1 selects a configuration; a **fresh complete pipeline refits on training plus
validation**, then predicts the untouched outer test partition once. Original audit
documents/manifests are preserved as historical artifacts. Saved configurations use
protocol ID `capture_cv_train_validation_refit_v1` to distinguish these results.

All validation-based decisions for the requested model and budgets are completed
before any outer test prediction. All final refits also complete before test scoring.
No estimator, imputer or scaler fitted in one outer round is reused in another.
Validation rows enter fitting only after selection is frozen. No outer test rows
ever enter a fit. Do not tune grids, seeds, feature lists or packet budgets using
test results from another manually run model. Descriptive EDA has already viewed
the population; this is not an untouched external benchmark.

## Shared data and feature policy

The loader reads the existing labeled flows/features, readiness cohort/feature audit,
and capture/flow/observation manifests. It selects `outer4_inner_holdout` only and
rejects inconsistent IDs, labels, provenance, missing N observations, partition
overlap, missing classes, changed primary features, and any documented recording
group crossing partitions. The manual research runner additionally requires exactly
851 common-cohort flows and 35 captures. It never regenerates labels or splits.
All four N representations use the same partition assignments. Unknown session
groups are never replaced by synthetic IDs. This is capture-disjoint weak-label
evaluation, not verified recording-independent activity recognition.

Fixed class order: CHAT, MAIL, VOIP, FT. UNRESOLVED is never a supervised target.
Predictors, in order: mean_length, std_length, min_length, max_length, forward_ratio,
mean_interarrival, std_interarrival. Lengths retain source byte units and IATs source
seconds. No identifiers, application names, protocol, observation delay, completed
flow statistics or provenance fields become predictors. Numeric selection is explicit.

Missing predictor cells use median imputation fitted inside each candidate's training
pipeline; entirely missing training columns and infinities fail rather than silently
changing the feature schema. Labels and split metadata cannot be imputed. The current
research data has no missing predictors; this policy supports future reviewed inputs.

## Independent model modules

All three use eight documented candidate configurations, seed 42 by default, no
class weighting or resampling. Tie-break: largest validation macro-F1, then lexical
order of canonical sorted-key parameter JSON. No test scores affect selection.

- Logistic Regression: SimpleImputer, StandardScaler, LogisticRegression(lbfgs,
  max_iter=2000). Multiclass handled by lbfgs with the four original labels. C values:
  .001, .01, .1, .3, 1, 3, 10, 100. Convergence warnings fail the run for review.
- Random Forest: SimpleImputer, RandomForestClassifier. Product grid: n_estimators
  [100,200], max_depth [None,10], min_samples_leaf [1,3]. No scaler, n_jobs=1.
- XGBoost: SimpleImputer, EncodedXGBClassifier wrapper. Original labels map to
  0..3 in the fixed class order internally and map back for every report. Product:
  n_estimators [100,200], max_depth [3,5], learning_rate [.05,.1]. Objective
  multi:softprob, num_class=4, tree_method=hist, device=cpu, n_jobs=1. No early
  stopping or test eval_set. Optional backend is imported only for its own execution.

The CPU device option requires XGBoost >=2.0; see the
[official parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).
No packages are installed automatically. Install manually if needed:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml-optional.txt
```

## Commands — training commands are for later manual approval/execution

No training happens on import, `--help`, or `--validate-only`:

```powershell
python -m src.ml.run_experiments --help
python -m src.ml.run_experiments --model logistic_regression --validate-only
python -m unittest discover -s tests -p test_ml_framework.py -v
```

After code review, run one model at a time (these commands train):

```powershell
python -m src.ml.run_experiments --model logistic_regression
python -m src.ml.run_experiments --model random_forest
python -m src.ml.run_experiments --model xgboost
```

Each command covers all four N and four outer rounds. A debug run may select one N:

```powershell
python -m src.ml.run_experiments --model logistic_regression --n 3
```

A single-N run still uses four folds; it is a real experiment, not a dry run. Output
paths are fixed per model, and an existing output directory causes refusal. Archive
previous results manually before switching from a debug run to a complete run.
Never treat debug/test scores as a way to choose configurations. `--root PATH`
selects an explicit project root, and `--seed` records a predeclared alternative.

Only after model results exist:

```powershell
python -m src.ml.compare_models
```

Comparison never retrains. Missing models produce a warning; zero results produce
no files. Partial comparisons are identified. Existing comparisons are not overwritten.
Different hashes, protocol versions, budgets, seeds, labels, predictors, outer-test
identities or fold assignments cause rejection, not silently unmatched comparisons.

## Future generated outputs (none generated during implementation)

Separate directories: results/ml/logistic_regression/, random_forest/, xgboost/.
Each will contain predictions.csv (test identities and original labels), fold_metrics,
fold_class_metrics, fold_confusion, pooled_metrics, pooled_class_metrics,
pooled_confusion, fold_metric_summary, selected_parameters, validation_search,
fit_events and training_class_roles CSVs, plus configuration.json and completed.json.
The completion marker is written last; partial runs are rejected by comparisons.
Source hashes, code hashes, package versions, class ordering, grid and refit policy
are recorded. No trained-model pickle is written; reproduction uses source and config.

Metrics: accuracy, macro/weighted F1, fixed-order per-class precision/recall/F1/support,
count and row-normalized confusion matrices. Zero predicted positives produce
precision=0 and an explicit flag. Missing true classes invalidate evaluation. Pooled
out-of-fold metrics are separated from mean/std fold metrics; folds are correlated,
so fold std is descriptive, not a confidence interval. Training-only majority/minority
roles use above/below the median of the four original training class counts (ties
middle), fixed across N; train+validation refit does not redefine those roles.

Comparison outputs are pooled/per-class CSVs, macro-F1 and per-class F1 plots at
300 dpi, and a limitations note. No fake metrics or placeholder performance artifacts.

## Scientific and implementation limits

Labels remain capture-derived; recording independence and archive-02 raw verification
remain unresolved. MAIL/FT have one test capture per round; sample count is not
independent-group count. Existing split similarity warnings remain relevant. The
fixed >=20 cohort estimates conditional long-flow performance, not all-flow reach.
This framework does not claim that minority classes need additional packets before
actual evaluation. Group-aware uncertainty intervals are not implemented in this
first runner; do not infer significance or generalization from pooled metrics alone.
Legacy src/ml/data.py, splits.py, train.py, evaluate.py, experiments.py and plots.py
remain historical stubs; the supported CLI and shared implementations are the new
modules described here. No protected datasets, labels, extraction code or split
manifests are modified by this implementation.
