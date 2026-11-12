"""Deterministic provisional capture-disjoint train/validation/test designs.

No model fitting. Capture grouping is not a claim of recording independence.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import pandas as pd

from src.analysis.dataset_eda import ROOT, BUDGETS, load_inputs, markdown
from src.analysis.ml_readiness import TARGETS, common_cohort, PRIMARY

PARTITIONS = ["train", "validation", "test"]
DESIGNS = {"outer4_inner_holdout": 4, "outer3_inner_holdout": 3, "single_holdout": 1}


def capture_counts(cohort):
    if cohort.flow_id.duplicated().any():
        raise ValueError("Duplicate cohort flow ID")
    if set(cohort.candidate_class) != set(TARGETS):
        raise ValueError("All four target classes are required")
    if cohort.groupby("capture_id").candidate_class.nunique().gt(1).any():
        raise ValueError("Capture has conflicting candidate classes")
    return cohort.groupby(["candidate_class", "capture_id"]).size().rename("flows").reset_index()


def subset_near(items, target, minimum, reserve_flows, reserve_captures=1):
    """Exhaustively choose a whole-capture subset by counts only, with stable ties."""
    items = sorted(items)
    best, chosen = None, None
    total = sum(count for _, count in items)
    for size in range(1, len(items) - reserve_captures + 1):
        for subset in combinations(items, size):
            count = sum(n for _, n in subset)
            if count < minimum or total - count < reserve_flows:
                continue
            key = (abs(count - target), size, tuple(cid for cid, _ in subset))
            if best is None or key < best:
                best, chosen = key, {cid for cid, _ in subset}
    if chosen is None:
        raise ValueError("No capture-disjoint allocation satisfies declared support screen")
    return chosen


def outer_buckets(items, folds):
    if len(items) < folds:
        raise ValueError("Too few captures for class-covered outer folds")
    buckets = [[] for _ in range(folds)]
    totals = [0] * folds
    for cid, count in sorted(items, key=lambda x: (-x[1], x[0])):
        index = min(range(folds), key=lambda j: (totals[j], len(buckets[j]), j))
        buckets[index].append(cid)
        totals[index] += count
    return buckets


def build_capture_assignments(cohort):
    counts = capture_counts(cohort)
    rows = []
    for design, folds in DESIGNS.items():
        for cls in TARGETS:
            sub = counts[counts.candidate_class == cls]
            items = list(zip(sub.capture_id, sub.flows.astype(int)))
            if design == "single_holdout":
                # 60/20/20 targets, subject to immutable whole captures and screens.
                total = sum(n for _, n in items)
                tests = [subset_near(items, total * .2, 20, 30, reserve_captures=2)]
            else:
                tests = [set(bucket) for bucket in outer_buckets(items, folds)]
            for round_index, test in enumerate(tests, 1):
                remaining = [(cid, n) for cid, n in items if cid not in test]
                target = (sum(n for _, n in items) * .2 if design == "single_holdout"
                          else sum(n for _, n in remaining) * .2)
                validation = subset_near(remaining, target, 10, 20)
                for cid, count in items:
                    partition = "test" if cid in test else "validation" if cid in validation else "train"
                    rows.append(dict(design=design, round=round_index, capture_id=cid,
                                     candidate_class=cls, partition=partition, flows=count,
                                     grouping_basis="capture_id_only_independence_unverified"))
    return pd.DataFrame(rows).sort_values(["design", "round", "capture_id"]).reset_index(drop=True)


def validate_assignments(cohort, features, captures, flows, observations):
    expected_ids = set(cohort.flow_id)
    expected_captures = set(cohort.capture_id)
    observed_rounds = set(zip(captures.design, captures["round"]))
    expected_rounds = {(d, r) for d, folds in DESIGNS.items() for r in range(1, folds + 1)}
    if observed_rounds != expected_rounds:
        raise ValueError("Unexpected design/round coverage")
    for design, round_index in sorted(expected_rounds):
        cap = captures[(captures.design == design) & (captures["round"] == round_index)]
        fl = flows[(flows.design == design) & (flows["round"] == round_index)]
        obs = observations[(observations.design == design) & (observations["round"] == round_index)]
        if cap.capture_id.duplicated().any() or set(cap.capture_id) != expected_captures:
            raise ValueError("Capture overlap or incomplete capture coverage")
        if fl.flow_id.duplicated().any() or set(fl.flow_id) != expected_ids:
            raise ValueError("Flow overlap or incomplete flow coverage")
        if not set(cap.partition) == set(PARTITIONS) or not fl.partition.isin(PARTITIONS).all():
            raise ValueError("Invalid or absent partition")
        mapping = cap.set_index("capture_id").partition
        if not fl.partition.eq(fl.capture_id.map(mapping)).all():
            raise ValueError("Flow crosses capture partition")
        if not cap.set_index("capture_id").candidate_class.equals(
                cohort.groupby("capture_id").candidate_class.first().reindex(cap.capture_id).set_axis(cap.capture_id)):
            raise ValueError("Capture target metadata differs from cohort")
        if len(obs) != 4 * len(cohort) or obs.duplicated(["flow_id", "n_packets"]).any():
            raise ValueError("Missing or duplicate N representations")
        if set(zip(obs.flow_id, obs.n_packets)) != {(fid, n) for fid in expected_ids for n in BUDGETS}:
            raise ValueError("Invalid observation identities")
        if not obs.partition.eq(obs.flow_id.map(fl.set_index("flow_id").partition)).all():
            raise ValueError("N representations cross partitions")
        for col in ["capture_id", "candidate_class"]:
            if not obs[col].eq(obs.flow_id.map(fl.set_index("flow_id")[col])).all():
                raise ValueError("Observation metadata differs from flow")
        actual_counts = fl.groupby("capture_id").size()
        if not cap.flows.eq(cap.capture_id.map(actual_counts)).all():
            raise ValueError("Capture counts differ from flow assignments")
        for partition in PARTITIONS:
            selected = fl[fl.partition == partition]
            if set(selected.candidate_class) != set(TARGETS):
                raise ValueError("Class missing from partition")
        joined = fl.merge(cohort[["flow_id", "capture_id", "candidate_class"]], on="flow_id", suffixes=("", "_source"), validate="one_to_one")
        for col in ["capture_id", "candidate_class"]:
            if not joined[col].eq(joined[col + "_source"]).all():
                raise ValueError("Manifest identity metadata differs from source")
    for design, folds in DESIGNS.items():
        if folds > 1:
            test = flows[(flows.design == design) & (flows.partition == "test")]
            if len(test) != len(cohort) or test.flow_id.duplicated().any():
                raise ValueError("Outer tests must cover every flow exactly once")
    # Match source observations, independently of manifest row order.
    valid = set(zip(features.flow_id, features.n_packets))
    if not set(zip(observations.flow_id, observations.n_packets)).issubset(valid):
        raise ValueError("Observation does not exist in source features")


def boundary_warnings(features, flows):
    """Same-N aggregate collisions only; no automatic removal or regrouping."""
    warnings = []
    sub = features[features.flow_id.isin(flows.flow_id.unique())].copy()
    for mode in ["exact", "coarsened"]:
        keys = sub[["n_packets", "protocol", *PRIMARY]].copy()
        if mode == "coarsened":
            for col in PRIMARY:
                keys[col] = keys[col].round(0 if "length" in col else 3)
        for _, group in sub.groupby([keys[col] for col in keys.columns], dropna=False, sort=True):
            if len(group) < 2:
                continue
            for (design, r), assigned in flows[flows.flow_id.isin(group.flow_id)].groupby(["design", "round"]):
                if assigned.partition.nunique() > 1:
                    warnings.append(dict(design=design, round=r, n_packets=int(group.n_packets.iloc[0]),
                                         comparison=mode, flows=len(assigned),
                                         captures=assigned.capture_id.nunique(),
                                         partitions=";".join(sorted(assigned.partition.unique())),
                                         candidate_classes=";".join(sorted(assigned.candidate_class.unique())),
                                         flow_ids=";".join(sorted(assigned.flow_id)),
                                         interpretation="aggregate match only; conversation duplication unverified"))
    return pd.DataFrame(warnings, columns=["design", "round", "n_packets", "comparison", "flows", "captures", "partitions", "candidate_classes", "flow_ids", "interpretation"])


def run(root=ROOT):
    root = Path(root)
    # Preserve all existing data, labeling, EDA and readiness outputs.
    protected = list((root / "data/processed").rglob("*.csv"))
    for directory in ["results/labeling", "results/eda", "results/ml_readiness"]:
        protected.extend((root / directory).rglob("*"))
    protected += [root / "src/traffic/label_captures.py", root / "results/dataset_inventory.csv"]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    _, all_flows, features = load_inputs(root)
    cohort = common_cohort(all_flows, features)
    existing = pd.read_csv(root / "results/ml_readiness/common_cohort_manifest.csv", keep_default_na=False, dtype={"flow_id": str})
    if set(existing.flow_id) != set(cohort.flow_id) or existing.flow_id.duplicated().any():
        raise ValueError("Readiness cohort differs from current inputs")
    for col in ["capture_id", "candidate_class"]:
        if not cohort.set_index("flow_id")[col].sort_index().equals(existing.set_index("flow_id")[col].sort_index()):
            raise ValueError("Readiness cohort metadata differs")
    captures = build_capture_assignments(cohort)
    columns = ["flow_id", "capture_id", "candidate_class", "verification_status", "label_source", "recording_group"]
    flows = captures.drop(columns="flows").merge(cohort[columns], on=["capture_id", "candidate_class"], validate="many_to_many")
    flows = flows.sort_values(["design", "round", "flow_id"]).reset_index(drop=True)
    observations = flows.merge(features[["flow_id", "n_packets"]], on="flow_id", validate="many_to_many")
    observations = observations.sort_values(["design", "round", "flow_id", "n_packets"]).reset_index(drop=True)
    validate_assignments(cohort, features, captures, flows, observations)
    counts = flows.groupby(["design", "round", "partition", "candidate_class"]).agg(
        flows=("flow_id", "size"), captures=("capture_id", "nunique")).reset_index()
    counts["flow_screen_minimum"] = counts.partition.map({"train": 20, "validation": 10, "test": 20})
    counts["passes_flow_screen"] = counts.flows >= counts.flow_screen_minimum
    counts["single_capture_warning"] = counts.captures < 2
    comparison = []
    for design, rounds in DESIGNS.items():
        sub = counts[counts.design == design]
        row = dict(design=design, rounds=rounds, all_classes_in_all_partitions=True,
                   all_count_screens_pass=bool(sub.passes_flow_screen.all()),
                   recording_independence="UNVERIFIED", capture_disjoint=True)
        for partition in PARTITIONS:
            part = sub[sub.partition == partition]
            row[f"minimum_per_class_{partition}_flows"] = int(part.flows.min())
            row[f"minimum_per_class_{partition}_captures"] = int(part.captures.min())
        comparison.append(row)
    warnings = boundary_warnings(features, flows)
    output = root / "results/capture_splits"
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(capture_partition_manifest=captures, flow_partition_manifest=flows,
                  observation_partition_manifest=observations, partition_class_counts=counts,
                  design_comparison=pd.DataFrame(comparison), cross_partition_similarity_warnings=warnings)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    write_report(output, tables)
    (output / "EVALUATION_PROTOCOL.md").write_text(Path(__file__).with_name("capture_split_protocol.md").read_text(encoding="utf-8"), encoding="utf-8")
    after = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    if hashes != after:
        raise RuntimeError("Protected sources changed")
    result = dict(cohort_flows=len(cohort), cohort_captures=cohort.capture_id.nunique(),
                  rounds=sum(DESIGNS.values()), no_training_executed=True,
                  protected_files_unchanged=True, protected_sha256=hashes,
                  algorithm="v1: count-only LPT outer; exhaustive whole-capture validation subset; lexical ties",
                  recommended_design="outer4_inner_holdout", status="PROVISIONAL_CAPTURE_DISJOINT_NOT_VERIFIED_SESSION_DISJOINT",
                  no_test_scores_used=True,
                  fit_isolation="protocol requirement only; no estimator or preprocessing fits executed",
                  pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file() and p.name != "validation.json"})
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["protected_sha256", "output_sha256"]}, indent=2))
    return tables


def write_report(output, tables):
    counts = tables["partition_class_counts"]
    warnings = tables["cross_partition_similarity_warnings"]
    warning_counts = warnings.groupby(["design", "comparison", "n_packets"]).size().rename("cross_partition_clusters").reset_index()
    report = """# Capture-disjoint train/validation/test feasibility

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

""" + markdown(tables["design_comparison"]) + "\n\n## Counts by design, round, partition and class\n\n" + markdown(counts)
    report += """

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

""" + (markdown(warning_counts) if len(warning_counts) else "No cross-partition aggregate matches found.")
    report += """

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
"""
    (output / "SPLIT_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
