# ML readiness audit

## Decision: CONDITIONALLY READY

Preliminary **capture-disjoint, weak-label** evaluation has count-feasible alternatives,
subject to explicit approval of that limited claim and the protocol below. The
single-validation/single-test-capture design fails the CHAT support screen. Reliable
independent-session generalization remains **BLOCKED**: recording relationships and
flow-level labels are unverified. No partitions or models are generated here.

## Independently verified population

13,314 unique flows; 8,286 feature records. All 1,107
N=20 flows have one representation at each N. The proposed supervised cohort has
851 flows in CHAT/MAIL/VOIP/FT, excluding 256 unresolved N=20 flows.
The original data retains every record; exclusion is explicit in the proposed manifest.

| candidate_class | flows | captures |
| --- | --- | --- |
| CHAT | 92 | 15 |
| MAIL | 247 | 4 |
| VOIP | 389 | 12 |
| FT | 123 | 4 |

| candidate_class | TCP | UDP |
| --- | --- | --- |
| CHAT | 80 | 12 |
| MAIL | 237 | 10 |
| VOIP | 336 | 53 |
| FT | 115 | 8 |

The largest full-dataset capture is `NonVPN-PCAPs-02/hangouts_audio4.pcapng`: 4,352
flows (32.69%). Its cohort impact is in
`sensitivity_feasibility.csv`. Class concentration by capture is in `capture_group_audit.csv`.

## Identity, segmentation, and leakage evidence

- **Confirmed:** `flows.py:get_flow_key` canonicalizes protocol and endpoint IP/port
  pairs, merging reverse directions. Direction follows the first observed source.
- **Confirmed:** `iter_flows` separates inactivity gaps greater than 60 seconds for
  TCP and UDP; FIN/RST/SYN do not delimit connections. One conversation can become
  multiple timeout records. Conversely reused tuples can merge separate connections.
- **Confirmed:** 16-hex-digit IDs hash capture ID, tuple, and segment sequence. IDs
  are stable across N, but changes to capture paths or segmentation can change them.
  No duplicate flow IDs or duplicate (flow_id,N) records passed input validation.
- **Plausible risk:** timeout segments, cross-file capture copies, retransmissions,
  or related recordings could span train and test. Endpoint tuples and absolute
  timestamps were not exported, so underlying-conversation overlap cannot be ruled out.
- **Confirmed:** aggregate matches are reported in `aggregate_similarity_audit.csv`.
  Exact means exact equality of protocol plus the seven primary predictors at the
  same N. Coarsened means length statistics rounded to 1 byte, ratios to .001,
  and IAT statistics to .001 seconds. These are similarity screens, not raw-packet
  duplicate detection. Rounded-bin boundaries can miss near neighbors. No records
  are dropped. Different labels may share short-prefix aggregates legitimately.
- **Confirmed:** filename/application/rule fields directly reveal candidate labels;
  they must never be predictors. IPs, ports and absolute timestamps are absent.
- **Plausible risk:** link-layer sizes, MTU, capture interface, timing environment,
  protocol composition and recording settings can act as capture fingerprints even
  after identifiers are removed. No model-based artifact dominance is established.

## Recording evidence

No independent recording groups are documented in the current manifest. Similar
names and a/b suffixes are relationship hints only, listed as such; no invented
groups are assigned. Application identity is not a session group.
The [official dataset description](https://www.unb.ca/cic/datasets/vpn.html) describes
controlled application captures, regular/VPN sessions and incidental browsing, but
does not establish independence for each local filename. See `docs/capture_labeling.md`.
The weak capture labels are confirmed metadata, not verified flow ground truth.

## Feature review

`features.py:extract_features` slices the first N accepted packets before computing
aggregates. Full-flow statistics are separated in the flow table. Primary allowlist:
mean_length, std_length, min_length, max_length, forward_ratio, mean_interarrival, std_interarrival. Every original column and absent endpoint fields are audited
in `feature_audit.csv`. Protocol is reserved for sensitivity analysis; redundant
volume/reverse-ratio predictors are excluded. Observation delay is prefix-safe but
reserved as an outcome, not falsely described as future leakage.
Prefix means and IATs still encode application behavior, which is intended; the
unresolved issue is whether they generalize beyond recording artifacts.

## Split feasibility (counts only; no final allocation)

| candidate_class | common_flows | capture_count | documented_recording_groups | unknown_group_captures | min_capture_support | max_capture_support | nonempty_train_val_test_options | screened_options | proposed_train_min | proposed_train_max | proposed_validation_min | proposed_validation_max | proposed_test_min | proposed_test_max | screen | verified_session_split | capture_disjoint_split | four_fold_capacity_min_test_flows | four_fold_capacity_max_test_flows | four_fold_capacity_min_train_flows | maximum_class_covered_capture_cv_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CHAT | 92 | 15 | 0 | 15 | 3 | 10 | 210 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | train>=20;validation>=10;test>=20; descriptive only; ranges not simultaneous | BLOCKED_missing_independence_evidence | INSUFFICIENT_FOR_SCREEN | 22 | 24 | 68 | 15 |
| MAIL | 247 | 4 | 0 | 4 | 32 | 94 | 12 | 12 | 70.000 | 177.000 | 32.000 | 94.000 | 32.000 | 94.000 | train>=20;validation>=10;test>=20; descriptive only; ranges not simultaneous | BLOCKED_missing_independence_evidence | COUNT_FEASIBLE_PENDING_APPROVAL | 32 | 94 | 153 | 4 |
| VOIP | 389 | 12 | 0 | 12 | 5 | 81 | 132 | 70 | 249.000 | 352.000 | 16.000 | 81.000 | 21.000 | 81.000 | train>=20;validation>=10;test>=20; descriptive only; ranges not simultaneous | BLOCKED_missing_independence_evidence | COUNT_FEASIBLE_PENDING_APPROVAL | 93 | 101 | 288 | 12 |
| FT | 123 | 4 | 0 | 4 | 23 | 44 | 12 | 12 | 46.000 | 77.000 | 23.000 | 44.000 | 23.000 | 44.000 | train>=20;validation>=10;test>=20; descriptive only; ranges not simultaneous | BLOCKED_missing_independence_evidence | COUNT_FEASIBLE_PENDING_APPROVAL | 23 | 44 | 79 | 4 |

The screen enumerates ordered pairs of one validation and one test capture per
class, assigning remaining captures to training. It checks train>=20, validation>=10,
test>=20 per class. These thresholds are declared descriptive screens, not power
calculations or reliability guarantees. Min/max ranges need not occur together.
All such allocations would be capture-disjoint by construction but might share
underlying recordings. CHAT has only 3–10 flows per capture, so no single test
capture can meet the 20-example screen. Combining test captures or four-fold outer
evaluation must be considered instead; the screen is not silently relaxed.
Four-fold capacity columns balance sizes into anonymous bins without saving capture
assignments. They illustrate support only, not a final allocation. Four class-covered
outer capture folds are count-feasible; neither session independence nor sufficient
precision is guaranteed. For MAIL/FT each fold would hold out just one capture.
No actual train/test counts are claimed because no final split has been selected.

## Sensitivities and remaining blockers

| analysis | candidate_class | flows | captures | status |
| --- | --- | --- | --- | --- |
| remove_largest_capture | CHAT | 92 | 15 | COUNT_AVAILABLE_independence_unverified |
| remove_largest_capture | MAIL | 247 | 4 | COUNT_AVAILABLE_independence_unverified |
| remove_largest_capture | VOIP | 308 | 11 | COUNT_AVAILABLE_independence_unverified |
| remove_largest_capture | FT | 123 | 4 | COUNT_AVAILABLE_independence_unverified |
| TCP_only | CHAT | 80 | 15 | COUNT_AVAILABLE_independence_unverified |
| TCP_only | MAIL | 237 | 4 | COUNT_AVAILABLE_independence_unverified |
| TCP_only | VOIP | 336 | 12 | COUNT_AVAILABLE_independence_unverified |
| TCP_only | FT | 115 | 4 | COUNT_AVAILABLE_independence_unverified |
| UDP_only | CHAT | 12 | 6 | SPARSE_support_below_20 |
| UDP_only | MAIL | 10 | 4 | SPARSE_support_below_20 |
| UDP_only | VOIP | 53 | 12 | COUNT_AVAILABLE_independence_unverified |
| UDP_only | FT | 8 | 4 | SPARSE_support_below_20 |
| NonVPN-PCAPs-01 | CHAT | 52 | 9 | COUNT_AVAILABLE_independence_unverified |
| NonVPN-PCAPs-01 | MAIL | 247 | 4 | COUNT_AVAILABLE_independence_unverified |
| NonVPN-PCAPs-01 | VOIP | 150 | 6 | COUNT_AVAILABLE_independence_unverified |
| NonVPN-PCAPs-01 | FT | 0 | 0 | NO_CLASS_SUPPORT |
| NonVPN-PCAPs-02 | CHAT | 40 | 6 | COUNT_AVAILABLE_independence_unverified |
| NonVPN-PCAPs-02 | MAIL | 0 | 0 | NO_CLASS_SUPPORT |
| NonVPN-PCAPs-02 | VOIP | 239 | 6 | COUNT_AVAILABLE_independence_unverified |
| NonVPN-PCAPs-02 | FT | 123 | 4 | COUNT_AVAILABLE_independence_unverified |

- Removing the largest capture and protocol/archive subsets are count-auditable;
  class support is shown above. Use identical identities across N within each subset.
- Application-family holdouts are domain-shift diagnostics only. MAIL has one
  unspecified email-client family and FT one FTPS family, so all-class unseen-family
  evaluation is infeasible. Capture-family grouping needs provenance before it can
  support an independence claim. Filename hints cannot silently become groups.
- Weak-label sensitivity requires independently reviewed labels or explicit
  contamination scenarios; no verified subset currently exists. Do not infer labels
  by ARFF row matching or by classifier confidence.
- Capture-artifact diagnostics may compare feature distributions by capture within
  class/protocol and, only after approval, predict capture identity or compare
  capture-held-out errors. Neither establishes independent sessions.
- 0 of 49 saved extraction
  signatures match current source bytes. Reconcile historical extraction provenance
  and sample raw reproductions before final scientific results. Archive 02 raw
  verification remains outstanding according to the project context.

## Before training versus reportable limitations

Before any training: approve capture-disjoint weak-label scope or recover verified
session groups; freeze cohort/feature allowlist/split rules; reject overlaps and
missing class support; approve small-group uncertainty limits. For reliable-session
claims, recover recording provenance and validate labels/background traffic first.
Before final results, reconcile extraction provenance. Transparency alone cannot
turn related recordings or weak labels into independent verified observations.
Small capture support, long-flow selection and dataset age must remain limitations.

## Reproduction

`python -m src.analysis.ml_readiness` regenerates audit artifacts only.
`python -m unittest discover -s tests -p test_ml_readiness.py -v` runs audit tests.
See EVALUATION_PROTOCOL.md for the proposed future design. No ML implementation is authorized by this audit.
