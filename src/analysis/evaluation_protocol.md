# Proposed evaluation protocol — approval required before training

> **Historical readiness proposal.** This records the design considered before
> final split allocation and model implementation. It is not the completed study's
> methods specification. Proposed models, repeated seeds, uncertainty analyses,
> and future sensitivities below must not be reported as executed experiments.
> See the [completed protocol](../../docs/experiment_protocol.md) and
> [final results index](../../results/ml/paper_figures/README.md). The generated copy
> under `results/ml_readiness/` remains an unchanged historical artifact.

## Scope and claim

Study N={3,5,10,20} packets using fixed flow identities. Current targets are
CHAT, MAIL, VOIP, FT **capture-derived candidates**, not verified flow labels.
UNRESOLVED is excluded from supervised comparisons explicitly, retained in coverage
and descriptive analyses, and never treated as a fifth supervised class. Zero-support
BROWSING/STREAMING/P2P are outside the trained task, not zero-performing classes.
No ARFF-to-flow joins. No model is trained by the readiness audit.

Current decision: CONDITIONALLY READY for explicitly approved preliminary
capture-disjoint weak-label experiments; BLOCKED for independent-session or verified
application-performance claims. Approval must specify which claim is intended.

## Primary population and predictors

Use `common_cohort_manifest.csv`: flows with all four representations and one of
the four assigned candidates. Current expected size is 851. Its four representations
are repeated measurements; never split or count them as independent examples.
The same flow identities, candidate classes and partition assignments apply at all N.
The estimand is performance conditional on retained long flows and weak labeling,
not performance on all network traffic. The >=20 criterion is retrospective cohort
selection, not a predictor available to an online N=3 classifier.

Primary predictor allowlist: mean_length, std_length, min_length, max_length,
forward_ratio, mean_interarrival, std_interarrival. Exclude every identifier,
application/label/provenance/group field, archive, full-flow statistic and eligibility
flag. Separate models per N; N is not a predictor. Do not use dataframe-wide numeric
selection. Exclude redundant total_bytes/backward_ratio. Protocol is prefix-safe but
reserved for a predeclared sensitivity because it strongly changes cohort composition.
Keep time_to_n as observation-delay outcome; timing predictors remain subject to
environment confounding. Its omission is a design choice, not proof of future leakage.

## Split preference and acceptance gate

1. Preferred: verified recording-session groups, including linked files and segments
   from the same underlying recording. Obtain source annotations or recording logs.
   Null groups prohibit claiming session-disjoint evaluation.
2. Provisional: capture ID as an observable grouping key, subject to explicit approval
   that this only establishes capture-disjointness. It protects against within-file
   flow/segment sharing but not related files, duplicate captures, shared hosts or
   recording settings. Never fill recording_group with capture_id silently.
3. If capture support or class support is insufficient, collect more independent
   recordings, restrict an explicitly redefined task, or defer modeling. No random
   flow split fallback. Application-level grouping alone cannot establish session
   independence and may remove an entire target class.

Before any fit, save and lock a separate approved split manifest and hash. Require:

- Exactly one partition per flow; no duplicate, missing or unknown flow IDs.
- Zero grouping-key overlap between every pair of partitions (including train versus
  validation); zero linked-recording overlap where links are known.
- Identical flow identities and partitions for all N; all expected representations.
- Every target in training and evaluation, with train/validation/test flow counts,
  capture/group counts, and per-capture concentration reported separately.
- Review same-N aggregate matches across partitions. These are collision warnings,
  not proof of duplicated packets. If underlying duplicates are established, keep
  them in one group or exclude by a documented rule fixed before scoring.
- Label, grouping and predictor checks pass before feature transformation.

The audit enumerates count-only three-way options (one capture validation and one
test per class); it does not choose one. Its 20/10/20 per-class count screen is only
a descriptive feasibility check. Do not present it as statistical sufficiency.
CHAT fails that particular screen because its captures each contribute only 3–10
common-cohort flows. Consider multiple held-out CHAT captures or grouped outer CV;
do not lower the screen silently. Anonymous four-bin capacity ranges in the audit
illustrate a CV alternative without defining any final capture assignments.
Train/test counts remain proposed ranges until an allocation is approved.

## Holdout versus grouped cross-validation

A single holdout is fragile with only four MAIL and four FT captures. Prefer a
predeclared four-fold capture-disjoint outer evaluation if provisional grouping is
accepted and every fold passes class-support checks. Balance capture assignment
within classes deterministically using counts, not model scores. Report each fold's
flow and capture support; stratification cannot overcome related recordings.

Within each outer training set, use at most three capture-disjoint inner folds if
each class remains represented. If infeasible, use fixed predeclared hyperparameters
instead of tuning. Do not replace grouped folds with flow-level folds. Freeze and
reuse outer/inner assignments across all N and model families. If reliable three-way
holdout support is approved instead, tune only using its training/validation data
and evaluate its locked test set once. Choose the evaluation design before training;
do not try holdout and CV and select whichever produces higher scores.

## Future models and preprocessing

Compare DummyClassifier (most_frequent), Decision Tree, and Random Forest using
the same eligible observations and predictor policy. Primary run preserves natural
within-cohort imbalance: no resampling and no class weights. Use fixed settings first
or an identical predeclared small tuning budget for each N. Select configurations by
inner-fold macro-F1 only, and record settings and all attempted trials.

Validate numeric schemas, finite values and units before fitting. Investigate missing
values rather than silently dropping rows; if imputation is justified, learn medians
only on the current training fold, inside a pipeline. Trees need no scaling; any
secondary scaler, feature selection or dimensionality reduction is fit within the
training fold only. Never select features using all-data correlations with targets.

Class-weighted variants are secondary and derive weights from current training data
only. Oversampling is optional secondary work, performed inside training folds after
group assignment, never before splitting and never on validation/test rows. It does
not create new independent recording support. Do not oversample the primary study.

Fix partition seed 42 and predeclare model seeds [42,43,44,45,46] if repeated fits are
used. Log software versions, input/cohort/split hashes and seeds. Different model
seeds are not independent test datasets. Do not search seeds for a favorable result.

Define majority/minority **within the training partition only**: classes above the
median of the four training class counts are majority; below are minority; exact ties
are a separate middle category. Keep assignments fixed across N within that outer
partition. Do not reuse full-dataset or test class frequencies for this definition.
Report actual training counts and labels, allowing roles to change between folds.

## Outcomes and uncertainty

Primary: macro-F1 and per-class precision, recall, F1 and support. Fixed class order
[CHAT, MAIL, VOIP, FT]; include count and row-normalized confusion matrices. Report
zero predicted positives with precision=0 and explicit undefined flags; missing true
classes invalidate the planned fold rather than being hidden in a macro average.
Weighted F1/accuracy are secondary. Always show per-class outcomes alongside
majority/minority summaries; weighting choices for those summaries must be declared.

Estimate paired changes versus N=3 on the same held-out flows, model family and
partition. Predeclare comparisons and report all N; never pick the best packet budget
or model from test scores. Observation delay and coverage are separate outcomes.

Use group-level resampling for uncertainty: paired cluster bootstrap of held-out
verified sessions (or explicitly provisional captures), carrying all flows and all N
from a sampled group together. Preserve class representation, e.g. resample captures
within class when groups are class-pure. A nominal 95% percentile interval with 2,000
replicates may be reported only with group counts and a small-group warning.
Four MAIL/FT captures give very limited uncertainty resolution. Do not imply precise
coverage guarantees. If groups cross classes, use a method preserving that structure
and report replicates lacking classes; do not independently resample correlated rows.

For outer CV, report per-fold metrics and pooled out-of-fold confusion counts;
pooled F1 and mean fold F1 are distinct estimands. Folds share training data and are
not independent replicates for a t-test. Group-bootstrap out-of-fold predictions
describes observed-test composition uncertainty, not all training-set variation.
Report model-seed and fold variation separately; avoid naive row-level confidence
intervals or significance claims. Multiple comparisons require a predeclared correction
if hypothesis tests are attempted; exploratory contrasts should be labeled as such.

## Secondary populations and sensitivities

N-specific eligible cohorts describe operational reach. Maintain the same approved
capture/group assignment, adding eligible shorter flows only to their source group's
partition. Report class support and denominator per N. Retraining on changing cohorts
mixes a sample-size/composition effect with packet-budget effects; never substitute
it for the fixed-common-cohort primary comparison.

Report coverage within each candidate and all flows, including unresolved traffic;
observation delays in milliseconds with median/p90/p95 and sample counts. Inference
latency and feature-extraction cost need separate future measurements. Do not score
unknown/short flows as classifier errors unless a separately defined rejection policy
is evaluated. No claim that minority classes need more packets precedes actual results.

Predeclare sensitivity populations using `sensitivity_feasibility.csv`:

- Remove the largest capture; report resulting supports and fixed shared identities.
- TCP/UDP strata, only if every intended evaluated class has adequate support.
- Archive-specific populations; archive-transfer tests cannot evaluate classes absent
  from one side (MAIL is absent from folder 02 and FT from folder 01).
- Application-family holdouts as limited domain-shift diagnostics; all-class unseen-
  family evaluation is infeasible for single-family MAIL/FT in this subset.
- Independently reviewed weak-label subset or transparent contamination scenarios,
  never confidence-filtered pseudo-ground-truth. Background assignment needs evidence.
- Within-class/protocol feature comparisons by capture and possible future capture-
  predictability diagnostics; these are artifact checks, not proof of generalization.

## Required evidence before stronger claims

Recover independent recording provenance, resolve known file relationships, validate
background/flow labels, reconcile the saved extraction signatures with source code,
and sample raw regeneration, particularly archive 02. Preserve originals and report
every exclusion. The study remains conditional on long-flow selection, weak labeling,
limited recording diversity and historical capture conditions even after software checks.
