# Recommended explicit train/validation/test protocol

> **Historical split-design proposal.** Retained to explain the original feasibility
> analysis, not as the current training protocol. The completed experiments use
> `outer4_inner_holdout` with a fresh train+validation refit after training-only
> candidate selection. The no-refit policy, planned classifiers, and pre-training
> status below were superseded. See the [completed protocol](../../docs/experiment_protocol.md)
> and [ML reproduction guide](../../docs/ml_framework.md). The saved generated copy
> under `results/capture_splits/` is preserved with its original checksum.

## Status and intended claim

The saved designs establish capture-disjointness only. They do not verify recording
independence or flow-level ground truth. Targets remain capture-derived CHAT, MAIL,
VOIP and FT candidates. UNRESOLVED remains outside the supervised cohort and inside
descriptive coverage reporting. The original >=20-packet cohort is unchanged.

Recommend `outer4_inner_holdout`, subject to approval before model implementation.
Use the saved manifests, not newly randomized folds. Three-fold and single-holdout
manifests are feasibility alternatives, not opportunities to choose a higher score.
Grouping assumptions remain a limitation requiring recording provenance for stronger
claims. No models have been trained or outer test scores examined.

## Roles in each outer round

- **Training captures:** the only observations allowed in any `.fit` call during
  hyperparameter/model selection, including imputers, scalers, feature selectors,
  resampling and classifiers. Each fit uses the current N representation of those
  same training flows. Features follow the readiness audit allowlist.
- **Validation captures:** transformations use training-fitted objects only;
  predictions/metrics may choose hyperparameters or models using a fixed predeclared
  search budget. Never use validation rows in fitting preprocessing or classifiers
  during selection. No fitting on a combined train+validation table.
- **Outer test captures:** apply the frozen training-fitted pipeline once selection
  is complete. No test-driven feature choice, tuning, model choice, threshold choice,
  budget selection, seed selection, early stopping or label cleanup.

For this explicit-separation protocol, **do not refit on train+validation** before
test evaluation. Retain the chosen training-only fit or reproduce it on training
only with frozen settings. A future refit variant would be a separately approved
protocol, not a silent change. The input manifests never authorize access to test
labels for fitting or selection.

All N={3,5,10,20} use identical train/validation/test identities per round. Separate
models per N. Keep model families and tuning budgets comparable. Choose fixed
hyperparameters initially where validation support is weak; no tuning expansion
after viewing test performance. Dummy, Decision Tree and Random Forest are planned
comparators, not implemented here. Fix model seeds in advance, e.g. 42; never search
seeds. Allocation itself is deterministic and seed-free.

Complete validation-based choices for every outer round/N/model before running any
outer test scoring. Outer CV test folds are reused as training in other rounds by
design, but each round has separate estimators, transformations and selection state.
Never transfer fitted preprocessing, selected hyperparameters chosen from other
outer test scores, or validation predictions between rounds. Global exploratory EDA
has already viewed dataset distributions; disclose this and do not claim a wholly
unseen external test set. The protocol protects model-selection isolation, not
retroactively removing prior descriptive inspection.

## Support and uncertainty

Four outer folds hold out one MAIL and one FT capture in each round. Of their three
remaining captures, one validates and two train. Three-fold outer CV sometimes
leaves only one training capture per class; the single holdout has no robustness to
test allocation. No design creates additional independent recording evidence.

Report flow and capture counts per class and partition from partition_class_counts.csv.
The 20 training / 10 validation / 20 test flow screens are engineering checks, not
power calculations. Report one-capture warnings even if those screens pass. In
particular MAIL/FT class estimates from one test capture are highly capture-dependent.
If a flow screen fails, flag that design; do not discard classes or split captures.

Primary outcomes: per-class precision/recall/F1/support, macro-F1 and fixed-order
confusion matrices. Report each round and pooled out-of-fold predictions separately
from the mean fold metric. Compare N to N=3 on the same held-out flow identities.
Use fixed training-only majority/minority definitions from the readiness protocol;
roles remain unchanged across N within a round. Natural within-cohort imbalance is
the primary setting. No oversampling or weighting based on validation/test counts.

Repeated representations and within-capture flows are not independent replicates.
Group-level paired uncertainty across capture-held-out predictions may be reported
as provisional composition uncertainty, with small-group warnings. Four MAIL/FT
captures cannot support precise recording-generalization intervals. Do not use
naive flow bootstrap, independent-fold t-tests, or treat model seeds as independent
datasets. Observation delay and coverage remain separate from classification metrics.

## Overlap warnings and future safeguards

Review `cross_partition_similarity_warnings.csv` before model implementation.
Exact/coarsened aggregates do not establish duplicated conversations. Do not merge
groups based solely on similar feature vectors: identical short prefixes occur
naturally across classes. Raw endpoints/timestamps or provenance are needed to
resolve underlying-copy suspicions. If duplicates or linked recordings are confirmed,
regenerate all designs under a reviewed relationship policy before training.

The manifest validators enforce capture/flow/N isolation and class presence. They
cannot prove software fit isolation until a future ML runner exists. That runner
must test fit-input IDs (including preprocessing), isolate test-label access, log
training/validation selection history, and prove all fitted objects see training
rows only. Approval of the split protocol is not approval to train yet.

## Reproduction and files

```powershell
python -m src.analysis.capture_splits
python -m unittest discover -s tests -p test_capture_splits.py -v
```

`capture_partition_manifest.csv`: each capture's role and size per round.
`flow_partition_manifest.csv`: each flow's role plus weak-label provenance.
`observation_partition_manifest.csv`: each (flow,N) role, without feature values.
`partition_class_counts.csv`: explicit training/validation/test class support.
`design_comparison.csv`: minimum supports and feasibility flags.
`cross_partition_similarity_warnings.csv`: same-N matching-aggregate warnings.
`SPLIT_COMPARISON_REPORT.md`: design comparison and limitations.
`validation.json`: source/output hashes and reproducibility metadata.
