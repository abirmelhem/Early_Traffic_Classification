
# A7B — Dataset Consolidation
# Merge per-capture flow and feature CSVs into two datasets and combine validation reports.
# Verify identifiers, counts, and consistency without modifying the original files.

import csv
import os
from contextlib import ExitStack
from pathlib import Path

from config import PROCESSED_DIR, RESULTS_DIR, PACKET_BUDGETS


DATASET_DIR = RESULTS_DIR / "dataset"
VALIDATION_DIR = DATASET_DIR / "validation"
OUTPUT_DIR = PROCESSED_DIR

SOURCES = {
    "flows": DATASET_DIR / "flows",
    "features": PROCESSED_DIR / "captures",
}

OUTPUTS = {
    "flows": OUTPUT_DIR / "all_flows.csv",
    "features": OUTPUT_DIR / "all_features.csv",
    "summary": DATASET_DIR / "dataset_summary.csv",
    "issues": DATASET_DIR / "validation_issues.csv",
}


def load_table(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")

        return reader.fieldnames, list(reader)


def unique_by_capture(rows, source):
    result = {}

    for row in rows:
        capture_id = row["capture_id"]

        if not capture_id:
            raise ValueError(f"Empty capture ID in {source}")

        if capture_id in result:
            raise ValueError(
                f"Duplicate capture ID in {source}: {capture_id}"
            )

        result[capture_id] = row

    return result


def merge_capture_files(kind, summaries, stack):
    source_dir = SOURCES[kind]
    output_path = OUTPUTS[kind]
    temp_path = output_path.with_suffix(".csv.tmp")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_file = stack.enter_context(
        temp_path.open("w", newline="", encoding="utf-8")
    )

    writer = None
    columns = None
    total_rows = 0
    seen = set()
    counts = {}

    for capture_id, summary in summaries.items():
        filename = capture_id.replace("/", "__").replace("\\", "__")
        source_path = source_dir / f"{filename}.csv"

        if not source_path.is_file():
            raise FileNotFoundError(f"Missing file: {source_path}")

        count = 0
        budgets = {n: 0 for n in PACKET_BUDGETS}

        with source_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as source_file:
            reader = csv.DictReader(source_file)

            if not reader.fieldnames:
                raise ValueError(f"Missing header: {source_path}")

            if columns is None:
                columns = reader.fieldnames
                writer = csv.DictWriter(
                    output_file, fieldnames=columns
                )
                writer.writeheader()

            if reader.fieldnames != columns:
                raise ValueError(
                    f"Inconsistent columns: {source_path}"
                )

            for row in reader:
                if row["capture_id"] != capture_id:
                    raise ValueError(
                        f"Capture ID mismatch: {source_path}"
                    )

                flow_id = row["flow_id"]

                if not flow_id:
                    raise ValueError(
                        f"Empty flow ID: {source_path}"
                    )

                if kind == "features":
                    n = int(row["n_packets"])

                    if n not in budgets:
                        raise ValueError(
                            f"Unexpected packet budget: {n}"
                        )

                    key = (capture_id, flow_id, n)
                    budgets[n] += 1
                else:
                    key = (capture_id, flow_id)

                if key in seen:
                    raise ValueError(
                        f"Duplicate identifier: {key}"
                    )

                seen.add(key)
                writer.writerow(row)
                count += 1

        expected = int(
            summary["total_flows"]
            if kind == "flows"
            else summary["feature_rows"]
        )

        if count != expected:
            raise ValueError(
                f"{capture_id}: expected {expected} "
                f"{kind} rows, found {count}"
            )

        if kind == "features":
            for n in PACKET_BUDGETS:
                expected_n = int(summary[f"flows_n{n}"])

                if budgets[n] != expected_n:
                    raise ValueError(
                        f"{capture_id}: N={n}, "
                        f"expected {expected_n}, "
                        f"found {budgets[n]}"
                    )

        counts[capture_id] = count
        total_rows += count

    return temp_path, total_rows, counts


def combine_summary(summaries):
    _, coverage_rows = load_table(
        VALIDATION_DIR / "flow_coverage.csv"
    )

    _, filtering_rows = load_table(
        VALIDATION_DIR / "filtering_analysis.csv"
    )

    coverage = unique_by_capture(
        coverage_rows, "flow_coverage.csv"
    )

    filtering = unique_by_capture(
        filtering_rows, "filtering_analysis.csv"
    )

    expected_ids = set(summaries)

    if set(coverage) != expected_ids:
        raise ValueError("Flow coverage capture IDs do not match")

    if set(filtering) != expected_ids:
        raise ValueError("Filtering capture IDs do not match")

    rows = []

    for capture_id, summary in summaries.items():
        cover = coverage[capture_id]
        filter_row = filtering[capture_id]

        if int(summary["total_flows"]) != int(
            cover["total_flows"]
        ):
            raise ValueError(
                f"Flow total mismatch: {capture_id}"
            )

        if int(summary["total_packets"]) != int(
            filter_row["total_packets"]
        ):
            raise ValueError(
                f"Packet total mismatch: {capture_id}"
            )

        if int(summary["eligible_packets"]) != int(
            filter_row["eligible"]
        ):
            raise ValueError(
                f"Eligible packet mismatch: {capture_id}"
            )

        if int(summary["filtered_packets"]) != int(
            filter_row["filtered"]
        ):
            raise ValueError(
                f"Filtered packet mismatch: {capture_id}"
            )

        for n in PACKET_BUDGETS:
            if int(summary[f"flows_n{n}"]) != int(
                cover[f"flows_n{n}"]
            ):
                raise ValueError(
                    f"Budget mismatch: {capture_id}, N={n}"
                )

        combined = dict(summary)

        for key, value in cover.items():
            if key not in combined:
                combined[key] = value

        for key, value in filter_row.items():
            if key not in combined:
                combined[key] = value

        rows.append(combined)

    return rows


def write_table(path, rows, columns):
    temp_path = path.with_suffix(".csv.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)

    return temp_path


def main():
    _, summary_rows = load_table(
        DATASET_DIR / "processing_summary.csv"
    )

    summaries = unique_by_capture(
        summary_rows, "processing_summary.csv"
    )

    _, issue_rows = load_table(
        VALIDATION_DIR / "validation_issues.csv"
    )

    if issue_rows:
        raise ValueError(
            "Validation issues exist. Review them first."
        )

    combined_summary = combine_summary(summaries)

    print(f"Consolidating {len(summaries)} captures...")

    temporary_files = []

    try:
        with ExitStack() as stack:
            for kind in ("flows", "features"):
                temp_path, count, _ = merge_capture_files(
                    kind, summaries, stack
                )
                temporary_files.append(temp_path)
                print(f"{kind}: {count:,} rows")

        summary_temp = write_table(
            OUTPUTS["summary"],
            combined_summary,
            list(combined_summary[0])
        )

        temporary_files.append(summary_temp)

        issues_temp = write_table(
            OUTPUTS["issues"],
            issue_rows,
            ["capture_id", "issue"]
        )

        temporary_files.append(issues_temp)

        for kind in ("flows", "features", "summary", "issues"):
            target = OUTPUTS[kind]
            temp = target.with_suffix(".csv.tmp")
            os.replace(temp, target)

        print("\nCONSOLIDATION COMPLETED")

        for name, path in OUTPUTS.items():
            print(f"{name}: {path}")

        print("\nOriginal per-capture CSV files were preserved.")

    finally:
        for path in temporary_files:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
