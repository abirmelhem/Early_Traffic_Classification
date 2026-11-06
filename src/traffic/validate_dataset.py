
# A7 — Dataset Validation and Analysis
# Validate extracted flows, packet filtering, and first-N feature records.
# Identify data quality issues before application labeling and ML experiments.

import csv
from collections import Counter, defaultdict
from pathlib import Path

from config import RESULTS_DIR, PROCESSED_DIR, PACKET_BUDGETS
from src.traffic.inventory import discover_captures
from src.traffic.flows import FILTER_REASONS


DATASET_DIR = RESULTS_DIR / "dataset"
REPORT_DIR = DATASET_DIR / "validation"

SUMMARY_PATH = DATASET_DIR / "processing_summary.csv"
FLOW_DIR = DATASET_DIR / "flows"
FILTER_DIR = DATASET_DIR / "filtering"
FEATURE_DIR = PROCESSED_DIR / "captures"

COVERAGE_COLUMNS = [
    "capture_id",
    "total_flows",
    "short_flows",
    "short_flow_pct",
    *[f"flows_n{n}" for n in PACKET_BUDGETS],
    *[f"coverage_n{n}_pct" for n in PACKET_BUDGETS],
]

FILTER_COLUMNS = [
    "capture_id",
    "total_packets",
    "eligible",
    "filtered",
    "filtered_pct",
    *FILTER_REASONS,
]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def capture_paths(capture_id):
    name = capture_id.replace("/", "__").replace("\\", "__")

    return {
        "flows": FLOW_DIR / f"{name}.csv",
        "filtering": FILTER_DIR / f"{name}.csv",
        "features": FEATURE_DIR / f"{name}.csv",
    }


def validate_capture(summary):
    capture_id = summary["capture_id"]
    paths = capture_paths(capture_id)

    flows = read_csv(paths["flows"])
    filtering = read_csv(paths["filtering"])
    features = read_csv(paths["features"])

    issues = []
    flow_ids = set()
    flow_lengths = {}
    eligible_counts = Counter()
    feature_counts = Counter()

    for flow in flows:
        flow_id = flow["flow_id"]
        count = int(flow["packet_count"])

        if flow_id in flow_ids:
            issues.append("Duplicate flow ID")

        flow_ids.add(flow_id)
        flow_lengths[flow_id] = count

        if count < 1:
            issues.append("Flow with no packets")

        if (
            int(flow["forward_count"])
            + int(flow["backward_count"])
            != count
        ):
            issues.append("Flow direction counts mismatch")

        if flow["capture_id"] != capture_id:
            issues.append("Flow capture ID mismatch")

        for n in PACKET_BUDGETS:
            eligible = int(count >= n)
            eligible_counts[n] += eligible

            if int(flow[f"eligible_n{n}"]) != eligible:
                issues.append(f"Incorrect flow eligibility at N={n}")

    seen_features = set()

    for row in features:
        flow_id = row["flow_id"]
        n = int(row["n_packets"])
        key = (flow_id, n)

        if key in seen_features:
            issues.append("Duplicate feature row")

        seen_features.add(key)

        if flow_id not in flow_lengths:
            issues.append("Feature references unknown flow")
        elif flow_lengths[flow_id] < n:
            issues.append("Feature exceeds available packets")

        if n not in PACKET_BUDGETS:
            issues.append("Unexpected packet budget")
        else:
            feature_counts[n] += 1

        if row["capture_id"] != capture_id:
            issues.append("Feature capture ID mismatch")

    if len(flows) != int(summary["total_flows"]):
        issues.append("Total flow count mismatch")

    if len(features) != int(summary["feature_rows"]):
        issues.append("Total feature row count mismatch")

    for n in PACKET_BUDGETS:
        if eligible_counts[n] != int(summary[f"flows_n{n}"]):
            issues.append(f"Summary mismatch at N={n}")

        if feature_counts[n] != eligible_counts[n]:
            issues.append(f"Feature count mismatch at N={n}")

    filtering_counts = {
        row["reason"]: int(row["packet_count"])
        for row in filtering
    }

    total = int(summary["total_packets"])
    eligible = int(summary["eligible_packets"])
    filtered = int(summary["filtered_packets"])

    if eligible + filtered != total:
        issues.append("Packet totals mismatch")

    for name, expected in [
        ("total", total),
        ("eligible", eligible),
        ("filtered", filtered),
    ]:
        if filtering_counts.get(name) != expected:
            issues.append(f"Filtering report mismatch: {name}")

    reason_total = sum(
        filtering_counts.get(reason, 0)
        for reason in FILTER_REASONS
    )

    if reason_total != filtered:
        issues.append("Filtering reasons do not sum to filtered packets")

    short_flows = sum(
        int(flow["packet_count"]) < min(PACKET_BUDGETS)
        for flow in flows
    )

    total_flows = len(flows)

    coverage = {
        "capture_id": capture_id,
        "total_flows": total_flows,
        "short_flows": short_flows,
        "short_flow_pct": round(
            100 * short_flows / total_flows, 2
        ) if total_flows else 0,
    }

    for n in PACKET_BUDGETS:
        coverage[f"flows_n{n}"] = eligible_counts[n]
        coverage[f"coverage_n{n}_pct"] = round(
            100 * eligible_counts[n] / total_flows, 2
        ) if total_flows else 0

    filtering_row = {
        "capture_id": capture_id,
        "total_packets": total,
        "eligible": eligible,
        "filtered": filtered,
        "filtered_pct": round(
            100 * filtered / total, 2
        ) if total else 0,
        **{
            reason: filtering_counts.get(reason, 0)
            for reason in FILTER_REASONS
        },
    }

    return coverage, filtering_row, sorted(set(issues))


def save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries = read_csv(SUMMARY_PATH)
    discovered = {
        capture_id for _, capture_id in discover_captures()
    }

    summary_ids = [row["capture_id"] for row in summaries]
    coverage_rows = []
    filtering_rows = []
    issue_rows = []

    for capture_id, count in Counter(summary_ids).items():
        if count > 1:
            issue_rows.append({
                "capture_id": capture_id,
                "issue": "Duplicate summary record",
            })

    for capture_id in sorted(discovered - set(summary_ids)):
        issue_rows.append({
            "capture_id": capture_id,
            "issue": "Missing processing summary record",
        })

    for capture_id in sorted(set(summary_ids) - discovered):
        issue_rows.append({
            "capture_id": capture_id,
            "issue": "Capture not found in source folders",
        })

    for index, summary in enumerate(summaries, start=1):
        capture_id = summary["capture_id"]
        print(
            f"[{index}/{len(summaries)}] {capture_id}",
            flush=True,
        )

        try:
            coverage, filtering, issues = validate_capture(summary)
            coverage_rows.append(coverage)
            filtering_rows.append(filtering)

            for issue in issues:
                issue_rows.append({
                    "capture_id": capture_id,
                    "issue": issue,
                })

        except (OSError, ValueError, KeyError, csv.Error) as error:
            issue_rows.append({
                "capture_id": capture_id,
                "issue": str(error),
            })

    save_csv(
        REPORT_DIR / "flow_coverage.csv",
        coverage_rows,
        COVERAGE_COLUMNS,
    )

    save_csv(
        REPORT_DIR / "filtering_analysis.csv",
        filtering_rows,
        FILTER_COLUMNS,
    )

    save_csv(
        REPORT_DIR / "validation_issues.csv",
        issue_rows,
        ["capture_id", "issue"],
    )

    total_flows = sum(
        row["total_flows"] for row in coverage_rows
    )

    short_flows = sum(
        row["short_flows"] for row in coverage_rows
    )

    print("\nA7 — VALIDATION SUMMARY")
    print(f"Captures discovered: {len(discovered)}")
    print(f"Captures validated: {len(coverage_rows)}")
    print(f"Total flows: {total_flows}")
    print(f"Flows with fewer than 3 packets: {short_flows}")

    for n in PACKET_BUDGETS:
        count = sum(
            row[f"flows_n{n}"] for row in coverage_rows
        )

        percentage = (
            100 * count / total_flows if total_flows else 0
        )

        print(f"N={n}: {count} flows ({percentage:.2f}%)")

    print(f"Validation issues: {len(issue_rows)}")
    print(f"Reports saved: {REPORT_DIR}")


if __name__ == "__main__":
    main()
