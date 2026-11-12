"""Audit identities and proposed cohorts; never generate partitions or train models."""
import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.analysis.dataset_eda import ROOT, BUDGETS, load_inputs, markdown

TARGETS = ["CHAT", "MAIL", "VOIP", "FT"]
PRIMARY = ["mean_length", "std_length", "min_length", "max_length",
           "forward_ratio", "mean_interarrival", "std_interarrival"]


def common_cohort(flows, features):
    """Inputs must already pass load_inputs identity/metadata/alignment checks."""
    ids = set(features.groupby("flow_id").filter(
        lambda g: len(g) == 4 and set(g.n_packets) == set(BUDGETS)).flow_id)
    selected = flows[flows.flow_id.isin(ids) & flows.candidate_class.isin(TARGETS)].copy()
    return selected.drop(columns=["length_bin"], errors="ignore").sort_values("flow_id")


def feature_audit(flows, features):
    descriptions = {
        "flow_id": ("identifier", "hash of capture/key/segment assigned at flow creation", "yes", "EXCLUDE", "Identity only; encodes capture membership; not a numeric traffic predictor"),
        "capture_id": ("identifier", "capture filename/path", "yes", "EXCLUDE", "Filename derives candidate target directly"),
        "protocol": ("categorical", "TCP or UDP", "yes", "SECONDARY_ONLY", "Prefix-safe but strong class/capture confound; encode within folds"),
        "packet_count": ("full_flow", "completed flow packet count", "no", "EXCLUDE", "Future-packet information; cohort selection only"),
        "forward_count": ("full_flow", "completed forward packet count", "no", "EXCLUDE", "Future-packet information"),
        "backward_count": ("full_flow", "completed reverse packet count", "no", "EXCLUDE", "Future-packet information"),
        "duration_seconds": ("full_flow", "completed flow duration in seconds", "no", "EXCLUDE", "Future-packet information"),
        "n_packets": ("design", "observation budget", "yes", "EXCLUDE", "Train a separate model per N; constant within model"),
        "mean_length": ("numeric", "prefix mean captured packet size in bytes", "yes", "PRIMARY", "Prefix-only; capture link-layer artifacts remain possible"),
        "std_length": ("numeric", "prefix population size standard deviation in bytes", "yes", "PRIMARY", "Prefix-only; capture artifacts remain possible"),
        "min_length": ("numeric", "prefix minimum packet size in bytes", "yes", "PRIMARY", "Prefix-only; may encode framing artifacts"),
        "max_length": ("numeric", "prefix maximum packet size in bytes", "yes", "PRIMARY", "Prefix-only; MTU/application confounding possible"),
        "total_bytes": ("numeric", "sum of prefix packet sizes in bytes", "yes", "EXCLUDE_REDUNDANT", "N times mean_length at fixed N; no future leakage"),
        "forward_ratio": ("numeric", "prefix initial-source packet proportion", "yes", "PRIMARY", "Prefix-only; forward means first observed endpoint, not necessarily client"),
        "backward_ratio": ("numeric", "prefix reverse packet proportion", "yes", "EXCLUDE_REDUNDANT", "One minus forward_ratio; no future leakage"),
        "mean_interarrival": ("numeric", "mean consecutive prefix IAT in seconds", "yes", "PRIMARY", "Prefix-only; negative differences clipped upstream"),
        "std_interarrival": ("numeric", "population prefix IAT standard deviation in seconds", "yes", "PRIMARY", "Prefix-only; timing environment may identify captures"),
        "time_to_n": ("outcome", "prefix observation delay in seconds", "yes", "REPORT_ONLY", "Not future leakage; reserve as networking outcome; sensitivity inclusion predeclared"),
        "application": ("metadata", "filename-derived application", "yes", "EXCLUDE", "Direct target shortcut"),
        "candidate_class": ("target", "capture-derived unverified activity", "yes", "TARGET_ONLY", "Never a predictor; not verified flow truth"),
        "label_source": ("metadata", "rule and evidence URL", "yes", "EXCLUDE", "Rule reveals target"),
        "confidence": ("metadata", "candidate confidence category", "yes", "EXCLUDE", "Labeling metadata; not traffic evidence"),
        "verification_status": ("metadata", "label verification category", "yes", "EXCLUDE", "Provenance/filter only"),
        "recording_group": ("group", "documented recording identity if available", "unknown", "GROUP_ONLY", "All null currently; never infer from application or suffix"),
    }
    rows = []
    columns = sorted((set(flows.columns) | set(features.columns)) - {"archive", "length_bin"})
    for column in columns:
        if column.startswith("eligible_n"):
            spec = ("full_flow", "completed flow eligibility flag", "no", "EXCLUDE", "Future information; cohort bookkeeping only")
        else:
            if column not in descriptions:
                raise ValueError(f"Unaudited column: {column}")
            spec = descriptions[column]
        kind, meaning, prefix, recommendation, why = spec
        rows.append(dict(column=column, present_in=";".join(k for k, f in [("flows", flows), ("features", features)] if column in f),
                         kind=kind, meaning=meaning, available_at_n=prefix,
                         requires_later_packets=kind == "full_flow", recommendation=recommendation, justification=why))
    for column in ["source_ip", "destination_ip", "source_port", "destination_port", "absolute_timestamp", "filename"]:
        rows.append(dict(column=column, present_in="ABSENT", kind="identifier", meaning="not exported separately",
                         available_at_n="yes", requires_later_packets=False, recommendation="EXCLUDE",
                         justification="Endpoint/service/time identity can shortcut targets; missing fields also prevent provenance matching"))
    return pd.DataFrame(rows)


def capture_audit(flows, cohort, manifest):
    rows = []
    ids = sorted(flows.capture_id.unique())
    for cid in ids:
        sub = flows[flows.capture_id == cid]
        common = cohort[cohort.capture_id == cid]
        meta = manifest[manifest.capture_id == cid]
        if len(meta) != 1:
            raise ValueError("Manifest capture coverage is not one-to-one")
        for field in ["candidate_class", "application", "recording_group", "verification_status"]:
            if not sub[field].eq(meta.iloc[0][field]).all():
                raise ValueError(f"Manifest disagrees with flows: {field}")
        # This is a relationship hint, explicitly not an asserted recording group.
        stem = Path(cid).stem.lower()
        prefix = re.sub(r"\d+[ab]?$", "", stem).rstrip("_")
        related = [other for other in ids if other != cid and
                   re.sub(r"\d+[ab]?$", "", Path(other).stem.lower()).rstrip("_") == prefix]
        group = meta.iloc[0].recording_group
        rows.append(dict(capture_id=cid, candidate_class=sub.candidate_class.iloc[0],
                         application=sub.application.iloc[0], total_flows=len(sub),
                         n20_flows=int((sub.packet_count >= 20).sum()), supervised_common_flows=len(common),
                         recording_group=group, recording_group_evidence="not documented" if not group else "manifest entry; independent provenance review required",
                         possible_related_captures=";".join(related),
                         relationship_evidence="filename similarity only; NOT a session/group assignment",
                         full_dataset_share_pct=100 * len(sub) / len(flows),
                         within_common_class_pct=100 * len(common) / len(cohort[cohort.candidate_class == sub.candidate_class.iloc[0]]) if len(common) else np.nan))
    return pd.DataFrame(rows)


def split_feasibility(cohort):
    rows = []
    for cls in TARGETS:
        sub = cohort[cohort.candidate_class == cls]
        counts = sub.groupby("capture_id").size().to_numpy()
        # Count-only exhaustive screen: one capture validation, one capture test,
        # all remaining captures training. No partition identities are emitted.
        options = [(int(counts.sum()-v-t), int(v), int(t)) for v, t in permutations(counts, 2)]
        nonempty = [x for x in options if min(x) > 0]
        screened = [x for x in nonempty if x[0] >= 20 and x[1] >= 10 and x[2] >= 20]
        # Capacity illustration only: balance capture sizes into four anonymous
        # bins. No capture identities, randomized assignment, or split is saved.
        bins = [0] * 4
        for count in sorted(counts, reverse=True):
            bins[bins.index(min(bins))] += int(count)
        rows.append(dict(candidate_class=cls, common_flows=len(sub), capture_count=len(counts),
                         documented_recording_groups=sub.loc[sub.recording_group.ne(""), "recording_group"].nunique(),
                         unknown_group_captures=sub.loc[sub.recording_group.eq(""), "capture_id"].nunique(),
                         min_capture_support=int(min(counts)) if len(counts) else 0,
                         max_capture_support=int(max(counts)) if len(counts) else 0,
                         nonempty_train_val_test_options=len(nonempty), screened_options=len(screened),
                         proposed_train_min=min((x[0] for x in screened), default=None),
                         proposed_train_max=max((x[0] for x in screened), default=None),
                         proposed_validation_min=min((x[1] for x in screened), default=None),
                         proposed_validation_max=max((x[1] for x in screened), default=None),
                         proposed_test_min=min((x[2] for x in screened), default=None),
                         proposed_test_max=max((x[2] for x in screened), default=None),
                         screen="train>=20;validation>=10;test>=20; descriptive only; ranges not simultaneous",
                         verified_session_split="BLOCKED_missing_independence_evidence",
                         capture_disjoint_split="COUNT_FEASIBLE_PENDING_APPROVAL" if screened else "INSUFFICIENT_FOR_SCREEN",
                         four_fold_capacity_min_test_flows=min(bins),
                         four_fold_capacity_max_test_flows=max(bins),
                         four_fold_capacity_min_train_flows=int(counts.sum())-max(bins),
                         maximum_class_covered_capture_cv_folds=len(counts)))
    return pd.DataFrame(rows)


def duplicate_audit(features):
    rows = []
    # Same N only: representations across N are intentional repeated observations.
    for n in BUDGETS:
        sub = features[features.n_packets == n].copy()
        for mode in ["exact", "coarsened"]:
            keys = sub[["protocol", *PRIMARY]].copy()
            if mode == "coarsened":
                for col in PRIMARY:
                    decimals = 0 if "length" in col else 3
                    keys[col] = keys[col].round(decimals)
            tags = keys.apply(tuple, axis=1)
            for key, indexes in tags.groupby(tags).groups.items():
                members = sub.loc[indexes]
                if len(members) > 1:
                    rows.append(dict(n_packets=n, comparison=mode, matching_flows=len(members),
                                     captures=members.capture_id.nunique(), labels=members.candidate_class.nunique(),
                                     flow_ids=";".join(sorted(members.flow_id)),
                                     capture_ids=";".join(sorted(members.capture_id.unique())),
                                     interpretation="aggregate similarity; NOT proof of duplicated conversations"))
    return pd.DataFrame(rows, columns=["n_packets", "comparison", "matching_flows", "captures", "labels", "flow_ids", "capture_ids", "interpretation"])


def run(root=ROOT):
    root = Path(root)
    protected = list((root / "data/processed").rglob("*.csv")) + list((root / "results/labeling").glob("*")) + list((root / "results/eda").rglob("*"))
    protected += [root / "results/dataset_inventory.csv"] + list((root / "src/traffic").glob("*.py"))
    before = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    _, flows, features = load_inputs(root)
    for column in PRIMARY:
        values = pd.to_numeric(features[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite primary predictor: {column}")
    manifest = pd.read_csv(root / "results/labeling/capture_manifest.csv", keep_default_na=False)
    if set(manifest.capture_id) != set(flows.capture_id) or manifest.capture_id.duplicated().any():
        raise ValueError("Manifest and dataset capture sets differ")
    cohort = common_cohort(flows, features)
    audit = capture_audit(flows, cohort, manifest)
    feasible = split_feasibility(cohort)
    duplicates = duplicate_audit(features)
    support = cohort.groupby("candidate_class").agg(flows=("flow_id", "size"), captures=("capture_id", "nunique")).reindex(TARGETS, fill_value=0).reset_index()
    protocol = pd.crosstab(cohort.candidate_class, cohort.protocol).reindex(index=TARGETS, columns=["TCP", "UDP"], fill_value=0).reset_index()
    sensitivity = []
    largest = flows.capture_id.value_counts().index[0]
    for name, subset in [("remove_largest_capture", cohort[cohort.capture_id != largest]),
                         ("TCP_only", cohort[cohort.protocol == "TCP"]), ("UDP_only", cohort[cohort.protocol == "UDP"]),
                         *[(a, cohort[cohort.archive == a]) for a in sorted(flows.archive.unique())]]:
        for cls in TARGETS:
            sub = subset[subset.candidate_class == cls]
            sensitivity.append(dict(analysis=name, candidate_class=cls, flows=len(sub), captures=sub.capture_id.nunique(),
                                    status=("COUNT_AVAILABLE_independence_unverified" if len(sub) >= 20
                                            else "SPARSE_support_below_20" if len(sub) else "NO_CLASS_SUPPORT")))
    signatures = ["config.py", "src/traffic/parse_pcap.py", "src/traffic/flows.py", "src/traffic/features.py", "src/traffic/export.py"]
    digest = hashlib.sha256()
    for p in signatures: digest.update((root / p).read_bytes())
    statuses = [json.loads(p.read_text()) for p in (root / "results/dataset/status").glob("*.json")]
    output = root / "results/ml_readiness"
    output.mkdir(parents=True, exist_ok=True)
    tables = {"feature_audit": feature_audit(flows, features), "capture_group_audit": audit,
              "common_cohort_manifest": cohort, "split_feasibility": feasible,
              "common_cohort_support": support, "common_cohort_protocols": protocol,
              "aggregate_similarity_audit": duplicates, "sensitivity_feasibility": pd.DataFrame(sensitivity)}
    for name, table in tables.items(): table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    summary = dict(decision="CONDITIONALLY READY", reliable_session_generalization="BLOCKED",
                   unique_flows=len(flows), common_all_classes=int((flows.packet_count >= 20).sum()),
                   proposed_supervised_common_flows=len(cohort),
                   common_unresolved=int(((flows.packet_count >= 20) & flows.candidate_class.eq("UNRESOLVED")).sum()),
                   largest_capture=largest, largest_capture_flows=int(flows.capture_id.value_counts().iloc[0]),
                   captures=len(audit), documented_recording_groups=cohort.loc[cohort.recording_group.ne(""), "recording_group"].nunique(),
                   saved_pipeline_signatures_matching=sum(s["pipeline_hash"] == digest.hexdigest() for s in statuses),
                   extraction_status_files=len(statuses), no_partitions_generated=True,
                   primary_features=PRIMARY, protected_sha256=before)
    report = f"""# ML readiness audit

## Decision: CONDITIONALLY READY

Preliminary **capture-disjoint, weak-label** evaluation has count-feasible alternatives,
subject to explicit approval of that limited claim and the protocol below. The
single-validation/single-test-capture design fails the CHAT support screen. Reliable
independent-session generalization remains **BLOCKED**: recording relationships and
flow-level labels are unverified. No partitions or models are generated here.

## Independently verified population

{len(flows):,} unique flows; {len(features):,} feature records. All {summary['common_all_classes']:,}
N=20 flows have one representation at each N. The proposed supervised cohort has
{len(cohort):,} flows in CHAT/MAIL/VOIP/FT, excluding {summary['common_unresolved']} unresolved N=20 flows.
The original data retains every record; exclusion is explicit in the proposed manifest.

{markdown(support)}

{markdown(protocol)}

The largest full-dataset capture is `{largest}`: {summary['largest_capture_flows']:,}
flows ({100*summary['largest_capture_flows']/len(flows):.2f}%). Its cohort impact is in
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
{', '.join(PRIMARY)}. Every original column and absent endpoint fields are audited
in `feature_audit.csv`. Protocol is reserved for sensitivity analysis; redundant
volume/reverse-ratio predictors are excluded. Observation delay is prefix-safe but
reserved as an outcome, not falsely described as future leakage.
Prefix means and IATs still encode application behavior, which is intended; the
unresolved issue is whether they generalize beyond recording artifacts.

## Split feasibility (counts only; no final allocation)

{markdown(feasible)}

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

{markdown(tables['sensitivity_feasibility'])}

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
- {summary['saved_pipeline_signatures_matching']} of {len(statuses)} saved extraction
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
"""
    (output / "READINESS_REPORT.md").write_text(report, encoding="utf-8")
    template = Path(__file__).with_name("evaluation_protocol.md")
    (output / "EVALUATION_PROTOCOL.md").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    after = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    if before != after: raise RuntimeError("Protected inputs changed")
    summary["protected_files_unchanged"] = True
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "protected_sha256"}, indent=2))
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
