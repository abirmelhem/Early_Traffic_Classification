# Final paper figures

Analysis only: no model fitting, tuning, resplitting or dataset modification.

## Figure inventory

- `fig01_baseline_macro_f1.png` and `fig01_baseline_macro_f1.pdf`: Pooled baseline macro-F1 versus packet budget; three models.
- `fig02_methods_macro_f1.png` and `fig02_methods_macro_f1.pdf`: Pooled macro-F1 by training strategy, one panel per model.
- `fig03_majority_minority_f1.png` and `fig03_majority_minority_f1.pdf`: Mean pooled class F1: MAIL/VOIP majority and CHAT/FT minority; equal class weights.
- `fig04_logistic_regression_per_class_f1.png` and `fig04_logistic_regression_per_class_f1.pdf`: Pooled per-class F1 for Logistic Regression, all three strategies.
- `fig04_random_forest_per_class_f1.png` and `fig04_random_forest_per_class_f1.pdf`: Pooled per-class F1 for Random Forest, all three strategies.
- `fig04_xgboost_per_class_f1.png` and `fig04_xgboost_per_class_f1.pdf`: Pooled per-class F1 for XGBoost, all three strategies.
- `fig05_baseline_confusions.png` and `fig05_baseline_confusions.pdf`: Baseline RF/XGBoost at N=5 and N=20; counts and true-class row percentages.
- `fig06_fold_macro_f1.png` and `fig06_fold_macro_f1.pdf`: Four outer-fold macro-F1 trajectories for every model/strategy; descriptive variation, not confidence intervals.
- `fig07_acquisition_delay_coverage.png` and `fig07_acquisition_delay_coverage.pdf`: Acquisition delay for 851 fixed supervised flows and eligibility coverage of all 13,314 extracted flows.

PNG exports are 300 DPI; PDF exports retain vector charts and embedded TrueType fonts.
Lines connect observed budgets only; no smoothing or unsupported budget estimates are generated.

## Exact experiment inputs

- Logistic Regression / Baseline: `results/ml/logistic_regression/predictions.csv`
- Logistic Regression / Class weighting: `results/ml/weighted_exploratory/predictions.csv`
- Logistic Regression / Corrected ROS: `results/ml/imbalance_v2/logistic_regression/random_oversampling/predictions.csv`
- Random Forest / Baseline: `results/ml/random_forest/predictions.csv`
- Random Forest / Class weighting: `results/ml/weighted_exploratory/predictions.csv`
- Random Forest / Corrected ROS: `results/ml/imbalance_v2/random_forest/random_oversampling/predictions.csv`
- XGBoost / Baseline: `results/ml/xgboost/predictions.csv`
- XGBoost / Class weighting: `results/ml/imbalance_v2/xgboost/class_weighted/predictions.csv`
- XGBoost / Corrected ROS: `results/ml/imbalance_v2/xgboost/random_oversampling/predictions.csv`

Saved pooled/class/fold metrics and baseline/new confusion tables are cross-checked; `validation.json` lists every read input and its hash.
Historical `results/ml/oversampling_exploratory/` is never read. Historical LR/RF weighting has weaker execution provenance than the new runs.

## Aggregation and supporting tables

All model comparisons use the same 851 flows at every N and original four-fold test assignments. Class order is CHAT, MAIL, VOIP, FT; supports are 92, 247, 389, 123.
`pooled_metrics.csv` contains 36 pooled outer-test summaries. `pooled_class_metrics.csv` contains 144 precision/recall/F1/support rows.
`majority_minority.csv` averages the two pooled class F1 scores: MAIL/VOIP majority and CHAT/FT minority. Roles are verified against original training counts. These are not fold-averaged role scores.
`fold_metrics.csv`, `fold_class_metrics.csv` and `fold_summary.csv` retain individual folds and descriptive mean/sample SD/min/max. Fold SD/range is not a confidence interval; fold training pools overlap.
`pooled_confusions.csv` contains counts and percentages normalized within each true class.
`supervised_acquisition_delay.csv` validates `results/supervised_delay/supervised_delay_overall.csv` against `data/processed/all_features.csv` and the 851-flow cohort. Seconds are converted to milliseconds; quantiles use linear interpolation. Acquisition time excludes inference/processing latency.
`flow_coverage.csv` uses packet counts in `data/processed/all_flows.csv`; denominator is all 13,314 flows, including unresolved labels and short flows. Evaluation-cohort support remains 851 at every N.
`source_inventory.csv` records the nine approved source mappings. Identity/provenance checks use `results/ml_readiness/common_cohort_manifest.csv` and `results/capture_splits/flow_partition_manifest.csv`.

## Scientific limits

Capture-derived labels are unverified; captures are not proven independent sessions. MAIL/FT have four captures each and archive/class confounding. The cohort selects flows reaching 20 packets. Later balancing comparisons reuse inspected tests and frozen baseline parameters; observed differences are descriptive, not significance claims. Historical preprocessing provenance remains uncertain.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -B -m src.analysis.paper_figures
```

The command refuses an existing output directory. To regenerate while preserving all prior figures:

```powershell
.\.venv\Scripts\python.exe -B -m src.analysis.paper_figures --output results/ml/paper_figures/regenerated-01
```

Choose another new subdirectory for subsequent regenerations. No experimental outputs are overwritten.
