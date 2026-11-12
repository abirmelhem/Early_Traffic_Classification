# Capture-disjoint train/validation/test feasibility

## Decision

Recommend **four-fold outer capture CV with an explicit inner capture-disjoint
training/validation holdout**, conditionally on accepting weak labels and unverified
recording independence. This is preliminary capture-generalization evaluation only.
No models or transformations have been fit. None of these designs establishes
independent-session or verified-flow-label generalization. All three alternatives
are saved as count-based candidate designs; do not run all three and select by score.

Four-fold outer CV holds out each capture exactly once. MAIL and FT each have four
captures: each round has one test capture, one validation capture and two training
captures. This preserves more training capture diversity than the three-fold design,
where a two-capture outer test leaves only one training and one validation capture.
The single holdout gives more training flows but only one test allocation; held-out
capture identity could dominate its results. No class is removed.

## Allocation algorithm and declared screens

All assignments use class counts and capture IDs only, never features or model
performance. For outer CV, within each class, sort captures by decreasing cohort flow
count (lexical capture ID ties); place each in the currently smallest of K bins
(then capture count and bin index ties). Each bin is one outer test fold.
On remaining captures, enumerate all nonempty validation subsets, selecting the
closest to 20% of remaining class flows (then fewer captures, lexical ID ties),
while reserving training captures and requiring training>=20 and validation>=10
flows per class. The test screen is >=20 per class; failures are reported, never
silently fixed. For the single holdout, select test near 20% of all class flows,
then validation near 20% of all class flows; training receives the remainder.
Capture indivisibility prevents exact ratios. Screens are descriptive engineering
limits, not statistical adequacy guarantees. No randomness or seed search is used.

## Design comparison

| design | rounds | all_classes_in_all_partitions | all_count_screens_pass | recording_independence | capture_disjoint | minimum_per_class_train_flows | minimum_per_class_train_captures | minimum_per_class_validation_flows | minimum_per_class_validation_captures | minimum_per_class_test_flows | minimum_per_class_test_captures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer4_inner_holdout | 4 | True | True | UNVERIFIED | True | 54 | 2 | 14 | 1 | 22 | 1 |
| outer3_inner_holdout | 3 | True | True | UNVERIFIED | True | 44 | 1 | 12 | 1 | 30 | 1 |
| single_holdout | 1 | True | True | UNVERIFIED | True | 54 | 2 | 18 | 1 | 20 | 1 |

## Counts by design, round, partition and class

| design | round | partition | candidate_class | flows | captures | flow_screen_minimum | passes_flow_screen | single_capture_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer3_inner_holdout | 1 | test | CHAT | 30 | 5 | 20 | True | False |
| outer3_inner_holdout | 1 | test | FT | 44 | 1 | 20 | True | True |
| outer3_inner_holdout | 1 | test | MAIL | 94 | 1 | 20 | True | True |
| outer3_inner_holdout | 1 | test | VOIP | 128 | 3 | 20 | True | False |
| outer3_inner_holdout | 1 | train | CHAT | 50 | 8 | 20 | True | False |
| outer3_inner_holdout | 1 | train | FT | 56 | 2 | 20 | True | False |
| outer3_inner_holdout | 1 | train | MAIL | 121 | 2 | 20 | True | False |
| outer3_inner_holdout | 1 | train | VOIP | 209 | 7 | 20 | True | False |
| outer3_inner_holdout | 1 | validation | CHAT | 12 | 2 | 10 | True | False |
| outer3_inner_holdout | 1 | validation | FT | 23 | 1 | 10 | True | True |
| outer3_inner_holdout | 1 | validation | MAIL | 32 | 1 | 10 | True | True |
| outer3_inner_holdout | 1 | validation | VOIP | 52 | 2 | 10 | True | False |
| outer3_inner_holdout | 2 | test | CHAT | 32 | 5 | 20 | True | False |
| outer3_inner_holdout | 2 | test | FT | 33 | 1 | 20 | True | True |
| outer3_inner_holdout | 2 | test | MAIL | 83 | 1 | 20 | True | True |
| outer3_inner_holdout | 2 | test | VOIP | 132 | 5 | 20 | True | False |
| outer3_inner_holdout | 2 | train | CHAT | 48 | 8 | 20 | True | False |
| outer3_inner_holdout | 2 | train | FT | 67 | 2 | 20 | True | False |
| outer3_inner_holdout | 2 | train | MAIL | 132 | 2 | 20 | True | False |
| outer3_inner_holdout | 2 | train | VOIP | 205 | 4 | 20 | True | False |
| outer3_inner_holdout | 2 | validation | CHAT | 12 | 2 | 10 | True | False |
| outer3_inner_holdout | 2 | validation | FT | 23 | 1 | 10 | True | True |
| outer3_inner_holdout | 2 | validation | MAIL | 32 | 1 | 10 | True | True |
| outer3_inner_holdout | 2 | validation | VOIP | 52 | 3 | 10 | True | False |
| outer3_inner_holdout | 3 | test | CHAT | 30 | 5 | 20 | True | False |
| outer3_inner_holdout | 3 | test | FT | 46 | 2 | 20 | True | False |
| outer3_inner_holdout | 3 | test | MAIL | 70 | 2 | 20 | True | False |
| outer3_inner_holdout | 3 | test | VOIP | 129 | 4 | 20 | True | False |
| outer3_inner_holdout | 3 | train | CHAT | 50 | 8 | 20 | True | False |
| outer3_inner_holdout | 3 | train | FT | 44 | 1 | 20 | True | True |
| outer3_inner_holdout | 3 | train | MAIL | 94 | 1 | 20 | True | True |
| outer3_inner_holdout | 3 | train | VOIP | 208 | 6 | 20 | True | False |
| outer3_inner_holdout | 3 | validation | CHAT | 12 | 2 | 10 | True | False |
| outer3_inner_holdout | 3 | validation | FT | 33 | 1 | 10 | True | True |
| outer3_inner_holdout | 3 | validation | MAIL | 83 | 1 | 10 | True | True |
| outer3_inner_holdout | 3 | validation | VOIP | 52 | 2 | 10 | True | False |
| outer4_inner_holdout | 1 | test | CHAT | 23 | 4 | 20 | True | False |
| outer4_inner_holdout | 1 | test | FT | 44 | 1 | 20 | True | True |
| outer4_inner_holdout | 1 | test | MAIL | 94 | 1 | 20 | True | True |
| outer4_inner_holdout | 1 | test | VOIP | 97 | 2 | 20 | True | False |
| outer4_inner_holdout | 1 | train | CHAT | 55 | 9 | 20 | True | False |
| outer4_inner_holdout | 1 | train | FT | 56 | 2 | 20 | True | False |
| outer4_inner_holdout | 1 | train | MAIL | 121 | 2 | 20 | True | False |
| outer4_inner_holdout | 1 | train | VOIP | 233 | 9 | 20 | True | False |
| outer4_inner_holdout | 1 | validation | CHAT | 14 | 2 | 10 | True | False |
| outer4_inner_holdout | 1 | validation | FT | 23 | 1 | 10 | True | True |
| outer4_inner_holdout | 1 | validation | MAIL | 32 | 1 | 10 | True | True |
| outer4_inner_holdout | 1 | validation | VOIP | 59 | 1 | 10 | True | True |
| outer4_inner_holdout | 2 | test | CHAT | 22 | 3 | 20 | True | False |
| outer4_inner_holdout | 2 | test | FT | 33 | 1 | 20 | True | True |
| outer4_inner_holdout | 2 | test | MAIL | 83 | 1 | 20 | True | True |
| outer4_inner_holdout | 2 | test | VOIP | 98 | 3 | 20 | True | False |
| outer4_inner_holdout | 2 | train | CHAT | 56 | 10 | 20 | True | False |
| outer4_inner_holdout | 2 | train | FT | 67 | 2 | 20 | True | False |
| outer4_inner_holdout | 2 | train | MAIL | 132 | 2 | 20 | True | False |
| outer4_inner_holdout | 2 | train | VOIP | 234 | 7 | 20 | True | False |
| outer4_inner_holdout | 2 | validation | CHAT | 14 | 2 | 10 | True | False |
| outer4_inner_holdout | 2 | validation | FT | 23 | 1 | 10 | True | True |
| outer4_inner_holdout | 2 | validation | MAIL | 32 | 1 | 10 | True | True |
| outer4_inner_holdout | 2 | validation | VOIP | 57 | 2 | 10 | True | False |
| outer4_inner_holdout | 3 | test | CHAT | 23 | 4 | 20 | True | False |
| outer4_inner_holdout | 3 | test | FT | 23 | 1 | 20 | True | True |
| outer4_inner_holdout | 3 | test | MAIL | 38 | 1 | 20 | True | True |
| outer4_inner_holdout | 3 | test | VOIP | 101 | 3 | 20 | True | False |
| outer4_inner_holdout | 3 | train | CHAT | 55 | 9 | 20 | True | False |
| outer4_inner_holdout | 3 | train | FT | 77 | 2 | 20 | True | False |
| outer4_inner_holdout | 3 | train | MAIL | 177 | 2 | 20 | True | False |
| outer4_inner_holdout | 3 | train | VOIP | 231 | 7 | 20 | True | False |
| outer4_inner_holdout | 3 | validation | CHAT | 14 | 2 | 10 | True | False |
| outer4_inner_holdout | 3 | validation | FT | 23 | 1 | 10 | True | True |
| outer4_inner_holdout | 3 | validation | MAIL | 32 | 1 | 10 | True | True |
| outer4_inner_holdout | 3 | validation | VOIP | 57 | 2 | 10 | True | False |
| outer4_inner_holdout | 4 | test | CHAT | 24 | 4 | 20 | True | False |
| outer4_inner_holdout | 4 | test | FT | 23 | 1 | 20 | True | True |
| outer4_inner_holdout | 4 | test | MAIL | 32 | 1 | 20 | True | True |
| outer4_inner_holdout | 4 | test | VOIP | 93 | 4 | 20 | True | False |
| outer4_inner_holdout | 4 | train | CHAT | 54 | 9 | 20 | True | False |
| outer4_inner_holdout | 4 | train | FT | 77 | 2 | 20 | True | False |
| outer4_inner_holdout | 4 | train | MAIL | 177 | 2 | 20 | True | False |
| outer4_inner_holdout | 4 | train | VOIP | 237 | 7 | 20 | True | False |
| outer4_inner_holdout | 4 | validation | CHAT | 14 | 2 | 10 | True | False |
| outer4_inner_holdout | 4 | validation | FT | 23 | 1 | 10 | True | True |
| outer4_inner_holdout | 4 | validation | MAIL | 38 | 1 | 10 | True | True |
| outer4_inner_holdout | 4 | validation | VOIP | 59 | 1 | 10 | True | True |
| single_holdout | 1 | test | CHAT | 20 | 3 | 20 | True | False |
| single_holdout | 1 | test | FT | 23 | 1 | 20 | True | True |
| single_holdout | 1 | test | MAIL | 38 | 1 | 20 | True | True |
| single_holdout | 1 | test | VOIP | 78 | 3 | 20 | True | False |
| single_holdout | 1 | train | CHAT | 54 | 10 | 20 | True | False |
| single_holdout | 1 | train | FT | 77 | 2 | 20 | True | False |
| single_holdout | 1 | train | MAIL | 177 | 2 | 20 | True | False |
| single_holdout | 1 | train | VOIP | 233 | 6 | 20 | True | False |
| single_holdout | 1 | validation | CHAT | 18 | 2 | 10 | True | False |
| single_holdout | 1 | validation | FT | 23 | 1 | 10 | True | True |
| single_holdout | 1 | validation | MAIL | 32 | 1 | 10 | True | True |
| single_holdout | 1 | validation | VOIP | 78 | 3 | 10 | True | False |

`capture_partition_manifest.csv` reports every capture's flow count and partition.
`flow_partition_manifest.csv` assigns each of the 851 flows once per round.
`observation_partition_manifest.csv` expands each assignment to all four N values.
These metadata-only manifests do not rewrite source feature values or labels.

## Leakage and duplicate checks

Every round contains every cohort flow and capture exactly once; no capture or flow
crosses train/validation/test within that round. All four N representations retain
one partition. Each CV design tests each flow exactly once over its outer rounds.
Across rounds a capture may appropriately be training in one and testing in another;
models and preprocessing must be refit from scratch for each round.

Same-N exact/coarsened aggregate similarities are checked across boundaries below.
Coarsening uses length statistics rounded to whole bytes and ratios/IAT seconds to
three decimals, plus exact protocol. These are weak similarity screens, not proof of
duplicate packet conversations. No automatic deletion or relabeling occurs. Inspect
exact warnings before model implementation; actual conversation copies must be
linked or excluded under a reviewed rule and all designs regenerated. Missing
endpoints and absolute timestamps prevent ruling out cross-file conversation reuse.

| design | comparison | n_packets | cross_partition_clusters |
| --- | --- | --- | --- |
| outer3_inner_holdout | coarsened | 3 | 118 |
| outer3_inner_holdout | coarsened | 5 | 155 |
| outer3_inner_holdout | coarsened | 10 | 116 |
| outer3_inner_holdout | coarsened | 20 | 57 |
| outer3_inner_holdout | exact | 3 | 2 |
| outer4_inner_holdout | coarsened | 3 | 132 |
| outer4_inner_holdout | coarsened | 5 | 169 |
| outer4_inner_holdout | coarsened | 10 | 145 |
| outer4_inner_holdout | coarsened | 20 | 74 |
| single_holdout | coarsened | 3 | 24 |
| single_holdout | coarsened | 5 | 24 |
| single_holdout | coarsened | 10 | 27 |
| single_holdout | coarsened | 20 | 18 |

## Scientific limits and pre-training gate

MAIL/FT test and validation supports each derive from one capture; flow counts cannot
replace independent-group counts. CHAT support is spread over multiple small
captures. Capture-derived labels may include background traffic. All original
recording_group fields remain null. Separate captures may share a recording or
capture environment. Small numbers of held-out captures make precise class-specific
confidence intervals unreliable even when every class has enough flow rows.

Confirm this provisional claim and freeze exactly one evaluation design before
training. Recover recording relationships and resolve true duplicate conversations
for stronger claims. Validation/test fit isolation is specified in the protocol;
it cannot be empirically certified before an ML runner exists. No random flow split
fallback is permitted. Do not inspect outer test scores until all choices are fixed.

## Reproduction

```powershell
python -m src.analysis.capture_splits
python -m unittest discover -s tests -p test_capture_splits.py -v
```

`validation.json` records source/output hashes. Inputs, readiness and EDA outputs
are preserved. The generation algorithm and manifests are deterministic.
