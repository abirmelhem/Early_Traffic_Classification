# Completed experiment protocol

This document describes the implemented study. Earlier readiness and split proposals
are historical snapshots; the final protocol includes training-plus-validation
refitting and three classifiers. See the [ML guide](ml_framework.md) for grids and
execution details and the [final output index](../results/ml/paper_figures/README.md)
for numerical tables and figures.

## Population and features

The processed non-VPN subset contains 49 captures, 13,314 unique bidirectional flows,
and 8,286 flow-budget records. Thirty-five captures have unverified capture-derived
CHAT, MAIL, VOIP, or FT candidates; 14 remain unresolved. The fixed supervised cohort
contains 851 flows with all four N={3,5,10,20} representations (3,404 records):
CHAT=92, MAIL=247, VOIP=389, FT=123. Unresolved and short flows remain in full-population
coverage statistics. ARFF records are not matched to flows.

The seven predictors, in order, are `mean_length`, `std_length`, `min_length`,
`max_length`, `forward_ratio`, `mean_interarrival`, and `std_interarrival`.
Length units are bytes and interarrival units are seconds. No identifiers, protocol,
label metadata, archive, full-flow information, redundant prefix totals/ratios, or
`time_to_n` become predictors. Separate models are fitted per N.

## Partitioning and baseline selection

Use `outer4_inner_holdout` in the saved manifests under `results/capture_splits/`.
Training, validation, and test flow/capture IDs are disjoint within each round;
all budgets and model families reuse their assignments. Each flow appears in outer
testing once per N. Test-fold sizes are 258, 236, 185, and 172. Grouping is by capture,
not verified independent recording session. MAIL/FT each have two training, one
validation, and one test capture per round.

Baseline models are Logistic Regression, Random Forest, and CPU-only XGBoost.
Each model has eight candidate configurations. Every candidate fits imputation,
LR scaling where applicable, and the classifier on training rows only. Validation
macro-F1 chooses parameters; ties use lexical canonical parameter JSON. After all
selections are frozen, fresh complete pipelines refit on training plus validation.
Outer test data is used only for final prediction/scoring, not preprocessing fitting,
feature selection, weighting, resampling, or parameter selection. Model seed is 42.
All refits complete before outer test scoring within each invocation.

The baseline protocol ID is `capture_cv_train_validation_refit_v1`. The no-refit
choice in earlier split proposals was superseded before these baseline runs.
DummyClassifier and Decision Tree proposals were not part of the completed study.

## Exploratory balancing variants

Class weighting and corrected random oversampling reuse the original baseline-selected
parameters separately for each model/N/fold; neither independently retunes parameters.
Each variant fits a fresh pipeline on train+validation only. LR/RF weighting uses
balanced class weights. XGBoost uses per-row `M / (4 * M_class)` weights from that
fitting pool. Corrected ROS retains every original fitting observation and appends
replacement draws to the largest class size, using sorted IDs and seed `42 + fold`.
The sampling multiplicities are shared across budgets. ROS preprocessing fits the
resampled pool; weighted preprocessing fits the original combined pool. Validation
and test prediction inputs are never resampled. Strategies are not combined.

The final comparison uses baseline outputs, LR/RF `weighted_exploratory` outputs,
and four `imbalance_v2` experiments. The latter record protocol
`capture_cv_frozen_baseline_imbalance_v2`. Historical `oversampling_exploratory`
used bootstrap-style minority replacement that could discard originals and is
excluded from the conventional-ROS comparison. See [imbalance details](imbalance_experiments.md).
Balancing comparisons are exploratory because baseline tests were already inspected.

## Outcomes and interpretation

The final comparison contains 36 pooled configurations and 144 fold summaries.
Pooled metrics concatenate outer-test predictions for 851 flows. Fold means average
four separate scores equally; sample SD uses denominator three and is descriptive.
Do not interchange these aggregations or interpret fold SD as a confidence interval.
Saved metrics include accuracy, macro/weighted F1, per-class precision/recall/F1/support,
and count/row-normalized confusion matrices. No hypothesis tests or confidence
intervals are part of the final outputs.

Majority/minority roles use original training counts before balancing: above/below
the median class count. MAIL/VOIP are majority and CHAT/FT minority in every round.
Final group scores are equal-weight means of their two pooled class F1 values,
not binary classification scores or fold-averaged role scores.

Packet acquisition delay is `time_to_n`, converted from seconds to milliseconds,
on the fixed 851-flow supervised cohort. Full-population coverage is the number of
eligible unique flows divided by 13,314, including unresolved and short flows.
The earlier EDA also reports different populations; do not substitute its delay
summaries for supervised-cohort delays. Acquisition excludes inference latency.

## Scope of claims and reproduction

These are descriptive comparisons on a restricted, long-flow, weak-label cohort.
Capture separation does not verify recording independence or eliminate overfitting.
MAIL/FT have limited capture support and archive associations. Previously inspected
tests, historical extraction-signature mismatch, incomplete raw provenance, and
unpinned dependency ranges remain documented limitations. No causal imbalance,
universal packet optimum, deployed QoS, or complete raw-to-result claim is supported.

Use saved manifests rather than regenerating alternatives after viewing scores.
Reproduction commands and overwrite behavior are in the [ML guide](ml_framework.md).
Preserve original outputs and use an isolated copy for training or data regeneration.
