"""Conservative capture-derived candidates; never verified flow ground truth."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP", "P2P", "FT")
BUDGETS = (3, 5, 10, 20)
SOURCE = "https://www.unb.ca/cic/datasets/vpn.html"
METADATA = ("application", "candidate_class", "label_source", "confidence",
            "verification_status", "recording_group")
MANIFEST_COLUMNS = ("capture_id", *METADATA)
RULE_VERSION = "1"

# Full matches only: new or undocumented patterns fail closed.
RULES = (
    (r"aim_?chat_?\d+[ab]?", "AIM", "CHAT", "aim_chat"),
    (r"icq_?chat_?\d+[ab]?", "ICQ", "CHAT", "icq_chat"),
    (r"facebook_?chat_?\d+[ab]?", "Facebook", "CHAT", "facebook_chat"),
    (r"hangouts?_chat_\d+[ab]?", "Hangouts", "CHAT", "hangouts_chat"),
    (r"email\d+[ab]?", "Email client (unspecified)", "MAIL", "email"),
    (r"facebook_audio\d+[ab]?", "Facebook", "VOIP", "facebook_audio"),
    (r"hangouts_audio\d+[ab]?", "Hangouts", "VOIP", "hangouts_audio"),
    (r"ftps_(?:up|down)_\d+[ab]?", "FTPS", "FT", "ftps"),
)


def infer_capture(capture_id):
    """Return candidate metadata, with unknown independence represented by None."""
    filename = capture_id.replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(filename).stem.lower()
    result = dict(zip(METADATA, (
        None, "UNRESOLVED", "filename_rule_v1:undocumented; " + SOURCE,
        "unresolved", "unresolved", None,
    )))
    if Path(filename).suffix.lower() not in {".pcap", ".pcapng"}:
        return result
    for pattern, app, candidate, rule in RULES:
        if re.fullmatch(pattern, stem):
            result.update(application=app, candidate_class=candidate,
                          label_source=f"filename_rule_v1:{rule}; {SOURCE}",
                          confidence="capture_candidate_only",
                          verification_status="capture_derived_unverified")
            return result
    for pattern, app, reason in (
        (r"facebook_video\d+[ab]?", "Facebook", "ambiguous_video"),
        (r"hangouts_video\d+[ab]?", "Hangouts", "ambiguous_video"),
        (r"netflix\d+[ab]?", "Netflix", "undocumented_netflix"),
        (r"gmailchat\d+[ab]?", "Gmail chat", "undocumented_gmail_chat"),
    ):
        if re.fullmatch(pattern, stem):
            result.update(application=app,
                          label_source=f"filename_rule_v1:{reason}; {SOURCE}")
            break
    return result


def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Missing or duplicate columns: {path}")
        columns = list(reader.fieldnames)
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return columns, rows


def build_manifest(inventory):
    ids = [row["capture_id"] for row in inventory]
    if not ids or any(not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Inventory requires unique nonempty capture IDs")
    for row in inventory:
        if row["capture_id"].rsplit("/", 1)[-1] != row["filename"]:
            raise ValueError("Inventory filename disagrees with capture ID")
    return [{"capture_id": cid, **infer_capture(cid)} for cid in sorted(ids)]


def join_metadata(columns, rows, manifest):
    if set(columns) & set(METADATA):
        raise ValueError("Input already contains labeling metadata")
    mapping = {row["capture_id"]: row for row in manifest}
    if len(mapping) != len(manifest):
        raise ValueError("Duplicate manifest capture ID")
    joined = []
    for row in rows:
        cid = row["capture_id"]
        if cid not in mapping:
            raise ValueError(f"Unmapped capture: {cid}")
        joined.append({**row, **{k: mapping[cid][k] for k in METADATA}})
    return [*columns, *METADATA], joined


def validate_alignment(flows, features, manifest):
    mapping = {row["flow_id"]: row for row in flows}
    if len(mapping) != len(flows) or "" in mapping:
        raise ValueError("Duplicate or empty flow ID")
    manifest_ids = {row["capture_id"] for row in manifest}
    if {row["capture_id"] for row in flows} != manifest_ids:
        raise ValueError("Inventory and flow capture sets differ")
    expected = set()
    for row in flows:
        count = int(row["packet_count"])
        if count < 1:
            raise ValueError("Invalid packet count")
        for n in BUDGETS:
            if int(row[f"eligible_n{n}"]) != int(count >= n):
                raise ValueError("Eligibility flag disagrees with packet count")
            if count >= n:
                expected.add((row["flow_id"], n))
    actual = Counter()
    for row in features:
        fid = row["flow_id"]
        if fid not in mapping:
            raise ValueError("Feature references unknown flow")
        if any(row[k] != mapping[fid][k] for k in ("capture_id", "protocol")):
            raise ValueError("Feature metadata disagrees with flow")
        actual[(fid, int(row["n_packets"]))] += 1
    if set(actual) != expected or any(v != 1 for v in actual.values()):
        raise ValueError("Missing, duplicate, or unexpected feature representation")


def eligibility_summary(flows, manifest, min_n20):
    results = []
    for candidate in (*CLASSES, "UNRESOLVED"):
        captures = {r["capture_id"] for r in manifest if r["candidate_class"] == candidate}
        subset = [r for r in flows if r["capture_id"] in captures]
        # Inferred capture labels never establish recording independence.
        groups = {r["recording_group"] for r in manifest
                  if r["capture_id"] in captures and r["recording_group"] is not None}
        for n in BUDGETS:
            eligible = [r for r in subset if int(r["packet_count"]) >= n]
            n20 = sum(int(r["packet_count"]) >= 20 for r in subset)
            results.append(dict(
                candidate_class=candidate, n_packets=n, total_flows=len(subset),
                eligible_flows=len(eligible),
                coverage_pct=round(100 * len(eligible) / len(subset), 6) if subset else None,
                distinct_captures=len(captures),
                eligible_captures=len({r["capture_id"] for r in eligible}),
                documented_recording_groups=len(groups),
                captures_with_unknown_group=sum(r["recording_group"] is None for r in manifest
                                                if r["capture_id"] in captures),
                n20_support_status=("unresolved_not_a_class" if candidate == "UNRESOLVED"
                                    else "below_screening_threshold" if n20 < min_n20
                                    else "passes_count_screen_only"),
                group_split_status="BLOCKED_unknown_independence" if captures else "NO_DATA",
            ))
    return results


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)  # None is an empty CSV cell; JSON preserves explicit null.


def run(root=ROOT, min_n20=100):
    root = Path(root)
    if min_n20 < 1:
        raise ValueError("N20 screening threshold must be positive")
    sources = {"inventory": root / "results/dataset_inventory.csv",
               "flows": root / "data/processed/all_flows.csv",
               "features": root / "data/processed/all_features.csv"}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}
    _, inventory = read_table(sources["inventory"])
    flow_cols, flows = read_table(sources["flows"])
    feature_cols, features = read_table(sources["features"])
    manifest = build_manifest(inventory)
    validate_alignment(flows, features, manifest)
    labeled = {"flows": join_metadata(flow_cols, flows, manifest),
               "features": join_metadata(feature_cols, features, manifest)}
    summary = eligibility_summary(flows, manifest, min_n20)
    # All input validation precedes output creation. Never rewrite an original input.
    output = root / "data/processed/capture_labeled"
    reports = root / "results/labeling"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(reports / "capture_manifest.csv", MANIFEST_COLUMNS, manifest)
    (reports / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for kind, (columns, rows) in labeled.items():
        write_csv(output / f"all_{kind}_capture_labeled.csv", columns, rows)
    write_csv(reports / "class_by_n.csv", list(summary[0]), summary)
    unresolved = [r for r in manifest if r["candidate_class"] == "UNRESOLVED"]
    write_csv(reports / "unresolved_captures.csv", MANIFEST_COLUMNS, unresolved)
    after = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}
    if hashes != after:
        raise RuntimeError("Input files changed during labeling")
    report = dict(rule_version=RULE_VERSION, source=SOURCE, packet_budgets=BUDGETS,
                  input_sha256=hashes,
                  pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  n20_screening_threshold=min_n20,
                  threshold_note="Descriptive warning only, not statistical adequacy or split feasibility",
                  capture_count=len(manifest), flow_count=len(flows), feature_count=len(features),
                  unresolved_capture_count=len(unresolved),
                  original_inputs_unchanged=True, all_flow_representations_present=True,
                  verified_flow_labels=0, documented_recording_groups=0,
                  warning="Candidates describe capture context; background flows remain unidentified. "
                          "No ARFF joins or independent recording groups are inferred.")
    (reports / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--min-n20", type=int, default=100,
                        help="Descriptive count screen; not a scientific sufficiency guarantee")
    args = parser.parse_args()
    run(args.root, args.min_n20)


if __name__ == "__main__":
    main()
