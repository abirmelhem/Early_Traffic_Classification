"""Read and validate existing data/manifests without fitting or rewriting them."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

from .features import BUDGETS, CLASSES, DESIGN, PREDICTORS, predictor_frame

ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={"flow_id": str})


def unique(frame, keys, name):
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {name}")


def require_columns(frame, names, name):
    if not set(names).issubset(frame.columns):
        raise ValueError(f"Missing {name} columns: {sorted(set(names)-set(frame.columns))}")


@dataclass
class ResearchData:
    features: pd.DataFrame
    assignments: pd.DataFrame
    cohort: pd.DataFrame
    hashes: dict

    def partition(self, fold, n, role):
        if fold not in (1, 2, 3, 4) or n not in BUDGETS or role not in ("train", "validation", "test"):
            raise ValueError("Invalid fold, budget, or partition")
        ids = self.assignments.loc[(self.assignments["round"] == fold) &
                                   (self.assignments.partition == role), "flow_id"]
        frame = self.features[self.features.n_packets.eq(n)].set_index("flow_id", drop=False)
        return frame.loc[sorted(ids)].copy()


def load_data(root=ROOT, expected_flows=None, expected_captures=None):
    root = Path(root)
    paths = {
        "flows": root / "data/processed/capture_labeled/all_flows_capture_labeled.csv",
        "features": root / "data/processed/capture_labeled/all_features_capture_labeled.csv",
        "cohort": root / "results/ml_readiness/common_cohort_manifest.csv",
        "audit": root / "results/ml_readiness/feature_audit.csv",
        "assignments": root / "results/capture_splits/flow_partition_manifest.csv",
        "observations": root / "results/capture_splits/observation_partition_manifest.csv",
        "captures": root / "results/capture_splits/capture_partition_manifest.csv",
    }
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    tables = {name: read_csv(path) for name, path in paths.items()}
    source, features, cohort, audit = (tables[k] for k in ["flows", "features", "cohort", "audit"])
    provenance = ["capture_id", "candidate_class", "verification_status", "label_source", "recording_group"]
    require_columns(source, ["flow_id", "packet_count", *provenance], "flow")
    require_columns(features, ["flow_id", "n_packets", *PREDICTORS, *provenance], "feature")
    require_columns(cohort, ["flow_id", *provenance], "cohort")
    require_columns(audit, ["column", "recommendation"], "audit")
    if set(audit.loc[audit.recommendation.eq("PRIMARY"), "column"]) != set(PREDICTORS):
        raise ValueError("Feature audit disagrees with hardcoded primary whitelist")
    unique(source, ["flow_id"], "source flow")
    unique(features, ["flow_id", "n_packets"], "feature representation")
    unique(cohort, ["flow_id"], "cohort flow")
    for frame in [source, features, cohort]:
        if frame.flow_id.str.strip().eq("").any() or frame.capture_id.str.strip().eq("").any():
            raise ValueError("Empty identifiers")
    if not features.n_packets.isin(BUDGETS).all():
        raise ValueError("Unexpected packet budget")
    expected_ids = set(source.loc[(pd.to_numeric(source.packet_count, errors="raise") >= 20) &
                                 source.candidate_class.isin(CLASSES), "flow_id"])
    if set(cohort.flow_id) != expected_ids or set(cohort.candidate_class) != set(CLASSES):
        raise ValueError("Common cohort differs from source or lacks a required class")
    if expected_flows is not None and len(cohort) != expected_flows:
        raise ValueError("Unexpected research cohort size")
    if expected_captures is not None and cohort.capture_id.nunique() != expected_captures:
        raise ValueError("Unexpected research capture count")
    expected_keys = {(fid, n) for fid in expected_ids for n in BUDGETS}
    selected = features[features.flow_id.isin(expected_ids)].copy()
    if set(zip(selected.flow_id, selected.n_packets)) != expected_keys:
        raise ValueError("Missing N representation in common cohort")
    for other, name in [(source[source.flow_id.isin(expected_ids)], "source"), (selected, "features")]:
        check = other.merge(cohort[["flow_id", *provenance]], on="flow_id", suffixes=("", "_cohort"), validate="many_to_one")
        for col in provenance:
            if not check[col].eq(check[col + "_cohort"]).all():
                raise ValueError(f"Inconsistent {name} {col}")
    if not selected.verification_status.eq("capture_derived_unverified").all():
        raise ValueError("Unexpected label verification status")
    predictor_frame(selected)  # NaNs permitted for training-fitted imputation, infinities rejected.
    assignments = tables["assignments"]
    observations = tables["observations"]
    captures = tables["captures"]
    for frame, name in [(assignments, "assignment"), (observations, "observation"), (captures, "capture")]:
        require_columns(frame, ["design", "round", "partition", "capture_id", "candidate_class"], name)
    require_columns(assignments, ["flow_id", *provenance], "assignment")
    require_columns(observations, ["flow_id", "n_packets", *provenance], "observation")
    assignments = assignments[assignments.design.eq(DESIGN)].copy()
    observations = observations[observations.design.eq(DESIGN)].copy()
    captures = captures[captures.design.eq(DESIGN)].copy()
    for frame in [assignments, observations, captures]:
        if set(frame["round"]) != {1, 2, 3, 4} or not frame.partition.isin(["train", "validation", "test"]).all():
            raise ValueError("Missing outer fold or invalid partition")
    for fold in (1, 2, 3, 4):
        a = assignments[assignments["round"].eq(fold)]
        o = observations[observations["round"].eq(fold)]
        c = captures[captures["round"].eq(fold)]
        unique(a, ["flow_id"], "fold flow assignment")
        unique(o, ["flow_id", "n_packets"], "observation assignment")
        unique(c, ["capture_id"], "capture assignment")
        if set(a.flow_id) != expected_ids or set(zip(o.flow_id, o.n_packets)) != expected_keys:
            raise ValueError("Incomplete fold flow/N coverage")
        if set(c.capture_id) != set(cohort.capture_id):
            raise ValueError("Incomplete capture coverage")
        if a.groupby("capture_id").partition.nunique().gt(1).any():
            raise ValueError("Capture overlap between partitions")
        if not a.partition.eq(a.capture_id.map(c.set_index("capture_id").partition)).all():
            raise ValueError("Capture and flow partition mismatch")
        for frame, name in [(a, "assignment"), (o, "observation")]:
            check = frame.merge(cohort[["flow_id", *provenance]], on="flow_id", suffixes=("", "_cohort"), validate="many_to_one")
            for col in provenance:
                if not check[col].eq(check[col + "_cohort"]).all():
                    raise ValueError(f"Inconsistent {name} {col}")
        if not o.partition.eq(o.flow_id.map(a.set_index("flow_id").partition)).all():
            raise ValueError("N representations cross partitions")
        for role in ["train", "validation", "test"]:
            if set(a.loc[a.partition.eq(role), "candidate_class"]) != set(CLASSES):
                raise ValueError("Missing class from partition")
        # Honor documented recording links if future evidence populates them.
        known = a[a.recording_group.ne("")]
        if known.groupby("recording_group").partition.nunique().gt(1).any():
            raise ValueError("Known recording group crosses partitions")
    test = assignments[assignments.partition.eq("test")]
    if len(test) != len(cohort) or test.flow_id.duplicated().any():
        raise ValueError("Outer tests do not cover each flow exactly once")
    return ResearchData(selected, assignments, cohort, hashes)
