# Final documentation review

Reviewed 24 November 2026 for *Early Network Traffic Classification under Class
Imbalance*, Aya Abdallah and Abir Melhem, American University of Beirut.

## Scope and source hierarchy

This review updates Markdown documentation only. Source code, dependency files,
configuration, labels, data, split manifests, predictions, numerical results, figures,
and the paper are outside the edit scope. No research training or result generation
is part of the review. Existing unrelated work is preserved.

The current specification is [experiment_protocol.md](experiment_protocol.md).
Actual code and saved configurations take precedence over earlier proposals.
The authoritative numerical comparison is
[results/ml/paper_figures/](../results/ml/paper_figures/README.md), whose
`source_inventory.csv` maps the nine completed model/strategy combinations.

| Source | Status |
|---|---|
| `results/ml/{logistic_regression,random_forest,xgboost}/` | Three completed baselines |
| `results/ml/weighted_exploratory/` | Final LR/RF class-weighted predictions/metrics, selected by model |
| `results/ml/imbalance_v2/logistic_regression/random_oversampling/` | Final corrected LR ROS |
| `results/ml/imbalance_v2/random_forest/random_oversampling/` | Final corrected RF ROS |
| `results/ml/imbalance_v2/xgboost/class_weighted/` | Final weighted XGBoost |
| `results/ml/imbalance_v2/xgboost/random_oversampling/` | Final corrected XGBoost ROS |
| `results/ml/oversampling_exploratory/` | Superseded minority bootstrap that could omit original rows |
| `results/ml/final_imbalance_comparison/` | Earlier historical-ROS comparison, not final conventional ROS |

The literal brace notation above abbreviates three existing baseline directories;
it is not an executable filesystem path.

## Documentation inspected

All 18 pre-existing project Markdown files were read:

- `README.md`.
- `docs/capture_labeling.md`, `docs/data_dictionary.md`, `docs/experiment_protocol.md`,
  `docs/imbalance_experiments.md`, `docs/ml_framework.md`.
- `src/analysis/capture_split_protocol.md`, `src/analysis/evaluation_protocol.md`.
- `results/capture_splits/EVALUATION_PROTOCOL.md`,
  `results/capture_splits/SPLIT_COMPARISON_REPORT.md`.
- `results/eda/REPORT.md`, `results/labeling/REPORT.md`.
- `results/ml_readiness/EVALUATION_PROTOCOL.md`,
  `results/ml_readiness/READINESS_REPORT.md`.
- `results/ml/comparisons/README.md`,
  `results/ml/comparisons/research_analysis/REPORT.md`.
- `results/ml/paper_figures/README.md`,
  `results/ml/weighted_exploratory/README.md`.

Generated reports under `results/` are preserved byte-for-byte. They describe their
original analysis stage; some say no models had yet been trained or propose a
different no-refit protocol. Their checksums are part of saved validation evidence.
The current README/protocol explain their historical status without modifying the
artifacts or their recorded hashes. Earlier baseline analysis remains useful, but
its role summaries average folds first; final role summaries average pooled class F1.

## Documentation changed

| File | Change |
|---|---|
| `README.md` | Replace preliminary overview, inaccurate tree, placeholder clone instructions, and planned models with verified study, final results, artifact map, reproduction boundaries, authors, and limitations |
| `docs/experiment_protocol.md` | Replace preliminary bullet plan with implemented cohort, predictors, splitting, baseline selection/refit, balancing, aggregation, and outcome definitions |
| `docs/data_dictionary.md` | Correct actual predictor flags, weak-label output schema, packet ordering, delay semantics, cohort counts, and implemented split description |
| `docs/ml_framework.md` | Mark experiments completed, explain supported commands/output preservation, replace nonexistent legacy-module descriptions, and document earlier pipeline stages |
| `docs/imbalance_experiments.md` | Identify completed/final sources, separate historical ROS, document actual LR/RF weighting command and versioned-rerun distinction |
| `docs/capture_labeling.md` | Add completed-study context and clarify regeneration overwrites dedicated annotations |
| `src/analysis/capture_split_protocol.md` | Add historical-status notice identifying superseded no-refit choice |
| `src/analysis/evaluation_protocol.md` | Add historical-status notice distinguishing proposed from executed methods |
| `docs/documentation_review.md` | Create this review and evidence record |

## Numerical verification

Read-only calculations checked the following against current CSV records:

- 13,314 unique flows and 8,286 unique flow/budget records; 49 capture-manifest rows.
- Candidate captures: CHAT=15, MAIL=4, VOIP=12, FT=4, unresolved=14.
- Fixed supervised cohort: 851 flows, 35 captures; CHAT=92, MAIL=247, VOIP=389, FT=123.
- Eligibility at N=3/5/10/20: 3,199 / 2,209 / 1,771 / 1,107 flows.
- Nine combinations with 3,404 predictions each: 30,636 records total, no repeated
  test identity within an N, with original true labels, capture IDs, and fold assignments.
- All 36 pooled accuracy/macro-F1 entries, 144 pooled class F1 entries, and 144
  fold macro-F1 entries agree with saved predictions to numerical tolerance 1e-12.

Baseline macro-F1 agrees with the requested three-decimal table. RF and XGBoost
peak at N=5 under every strategy, while LR macro-F1 increases through N=20. Both
balancing variants improve LR macro-F1 at all budgets. Final pooled role means
show majority F1 above minority F1 in all 36 configurations. No discrepancy with
the requested core counts or findings was found.

Evidence paths include `data/processed/all_flows.csv`, `all_features.csv` in the
same directory, `results/labeling/capture_manifest.csv`,
`results/ml_readiness/common_cohort_manifest.csv`,
`results/capture_splits/flow_partition_manifest.csv`, and the final paper tables.
`supervised_acquisition_delay.csv` uses 851 flows; `flow_coverage.csv` uses all
13,314 flows. Both are under `results/ml/paper_figures/`.

## Command and preservation verification

Documented module names, arguments, dependencies, destination rules, and entry-point
guards were checked against source. No preprocessing, training, installation, test
suite, or figure-regeneration command was executed. Seven no-training validation
invocations passed: all three `src.ml.run_experiments --model MODEL --validate-only`
checks, and `src.ml.imbalance_experiments --model MODEL --method METHOD --validate-only`
for the four completed versioned combinations listed above. Every invocation returned
`validated_no_training`, 851 flows, and 35 captures. These checks did not construct
estimators, fit models, score predictions, or write results. They are distinct from
synthetic tests and research experiments.

Training commands are provided for an isolated reproduction copy. Baseline and
versioned imbalance commands refuse existing outputs. Earlier inventory, extraction,
consolidation, labeling, EDA, readiness, and split commands may replace dedicated
derived outputs. The publication generator requires a new directory below
`results/ml/paper_figures/`; it does not train models.

No Git commands are used. Instead of a Git diff, the review compares documentation
using pre-edit file hashes and checks filesystem snapshots for non-Markdown
changes. Ordinary project files use SHA-256; large raw captures, archives, and ARFF
files use size/mtime metadata. Environment, Git internals, and Python caches are
excluded from this project-artifact snapshot. This verifies changes introduced
during the review, not any pre-existing uncommitted work.

Final checks passed: 19 Markdown files, 52 relative links, and no broken relative
links or checked heading anchors. All 50 input hashes and 29 output hashes recorded
by the final publication analysis still match, as does its analysis-script hash.
The filesystem comparison found eight modified Markdown files and this one new
Markdown file, no deleted files, and no non-Markdown changes. All generated result
documents remain unchanged. No Git diff was obtained because no Git commands were run.

## Paper cross-check and remaining qualifications

The local `C:/Users/AUB/Downloads/CMPS342_IEEE_spacing_fixed.tex` was read only.
Its title/authors, cohort, seven predictors, baseline selection and refit,
balancing sources, rounded classification tables, acquisition delays, and coverage
agree with the documented core study. It is outside this repository and is not
modified or presented as a distributed repository artifact. Its `paper_figures/`
image references require packaging the existing PDFs under that folder in the
manuscript project; the repository holds them under `results/ml/paper_figures/`.
This check is not a bibliography or paper-compilation audit.

Submission qualifications remain: capture-derived weak labels, unknown recording
independence, four MAIL/FT captures, archive/class associations, retrospective
20-packet cohort selection, exploratory reuse of inspected tests, incomplete
historical weighted-run metadata, historical extraction-signature mismatch, incomplete
raw archive-02 provenance, and unpinned dependencies. Saved configurations record
the actual Python/package versions; minimum dependency ranges alone do not establish
exact reproduction. No inference-latency or deployed QoS measurement is reported.
The documentation supports these qualified descriptive findings, not complete
raw-to-result reproducibility or causal/significance claims.
