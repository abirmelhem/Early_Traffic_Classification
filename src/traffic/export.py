
# A6 — Dataset Export
# Process all PCAP/PCAPNG captures using memory-efficient flow extraction.
# Save flow statistics, filtering reports, and first-N packet features.

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from config import (
    BASE_DIR,
    RESULTS_DIR,
    PROCESSED_DIR,
    PACKET_BUDGETS,
    MAX_PREFIX,
    TIMEOUT,
)
from src.traffic.inventory import discover_captures
from src.traffic.flows import (
    iter_flows,
    statistics_row,
    STATISTICS_COLUMNS,
    FILTER_REASONS,
)
from src.traffic.features import (
    iter_feature_rows,
    FEATURE_COLUMNS,
)


FLOW_DIR = RESULTS_DIR / "dataset" / "flows"
FILTER_DIR = RESULTS_DIR / "dataset" / "filtering"
FEATURE_DIR = PROCESSED_DIR / "captures"
STATUS_DIR = RESULTS_DIR / "dataset" / "status"
SUMMARY_PATH = RESULTS_DIR / "dataset" / "processing_summary.csv"

SUMMARY_COLUMNS = [
    "capture_id",
    "total_packets",
    "eligible_packets",
    "filtered_packets",
    "total_flows",
    "feature_rows",
    *[f"flows_n{n}" for n in PACKET_BUDGETS],
]


def pipeline_signature():
    digest = hashlib.sha256()

    for filename in [
        "config.py",
        "src/traffic/parse_pcap.py",
        "src/traffic/flows.py",
        "src/traffic/features.py",
        "src/traffic/export.py",
    ]:
        digest.update((BASE_DIR / filename).read_bytes())

    return digest.hexdigest()


def output_paths(capture_id):
    safe_name = capture_id.replace("/", "__").replace("\\", "__")

    return {
        "flows": FLOW_DIR / f"{safe_name}.csv",
        "filtering": FILTER_DIR / f"{safe_name}.csv",
        "features": FEATURE_DIR / f"{safe_name}.csv",
        "status": STATUS_DIR / f"{safe_name}.json",
    }


def capture_signature(path, capture_id, pipeline_hash):
    info = path.stat()

    return {
        "capture_id": capture_id,
        "file_size": info.st_size,
        "modified_ns": info.st_mtime_ns,
        "pipeline_hash": pipeline_hash,
    }


def is_completed(paths, signature):
    if not all(path.is_file() for path in paths.values()):
        return False

    try:
        with paths["status"].open(encoding="utf-8") as file:
            saved = json.load(file)

        return all(
            saved.get(key) == value
            for key, value in signature.items()
        )
    except (OSError, ValueError):
        return False


def process_capture(path, capture_id, pipeline_hash):
    paths = output_paths(capture_id)
    signature = capture_signature(path, capture_id, pipeline_hash)

    if is_completed(paths, signature):
        print(f"Skipping completed: {capture_id}", flush=True)
        return "skipped"

    print(f"\nProcessing: {capture_id}", flush=True)

    for output_path in paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    paths["status"].unlink(missing_ok=True)

    temporary = {
        name: output_path.with_name(output_path.name + ".tmp")
        for name, output_path in paths.items()
    }

    counters = defaultdict(int)
    flow_count = 0
    feature_count = 0
    coverage = {n: 0 for n in PACKET_BUDGETS}

    try:
        with ExitStack() as stack:
            flow_file = stack.enter_context(
                temporary["flows"].open(
                    "w", newline="", encoding="utf-8"
                )
            )

            feature_file = stack.enter_context(
                temporary["features"].open(
                    "w", newline="", encoding="utf-8"
                )
            )

            flow_writer = csv.DictWriter(
                flow_file, fieldnames=STATISTICS_COLUMNS
            )

            feature_writer = csv.DictWriter(
                feature_file, fieldnames=FEATURE_COLUMNS
            )

            flow_writer.writeheader()
            feature_writer.writeheader()

            for flow in iter_flows(path, capture_id, counters):
                flow_writer.writerow(statistics_row(flow))
                flow_count += 1

                for n in PACKET_BUDGETS:
                    if flow["packet_count"] >= n:
                        coverage[n] += 1

                for row in iter_feature_rows(flow):
                    feature_writer.writerow(row)
                    feature_count += 1

        with temporary["filtering"].open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["reason", "packet_count"])

            for reason in FILTER_REASONS:
                writer.writerow([reason, counters.get(reason, 0)])

            writer.writerow(["eligible", counters["eligible"]])
            writer.writerow(["filtered", counters["filtered"]])
            writer.writerow(["total", counters["total"]])

        summary = {
            "capture_id": capture_id,
            "total_packets": counters["total"],
            "eligible_packets": counters["eligible"],
            "filtered_packets": counters["filtered"],
            "total_flows": flow_count,
            "feature_rows": feature_count,
            **{
                f"flows_n{n}": coverage[n]
                for n in PACKET_BUDGETS
            },
        }

        for name in ["flows", "filtering", "features"]:
            os.replace(temporary[name], paths[name])

        with temporary["status"].open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {**signature, "summary": summary},
                file,
                indent=2,
            )

        os.replace(temporary["status"], paths["status"])

    finally:
        for temp_path in temporary.values():
            temp_path.unlink(missing_ok=True)

    print(f"Packets: {counters['total']}")
    print(f"Eligible packets: {counters['eligible']}")
    print(f"Flows: {flow_count}")
    print(f"Feature rows: {feature_count}")
    print(f"Saved: {paths['features']}", flush=True)

    return "processed"


def save_summary(captures, pipeline_hash):
    rows = []

    for path, capture_id in captures:
        paths = output_paths(capture_id)
        signature = capture_signature(path, capture_id, pipeline_hash)

        if not is_completed(paths, signature):
            continue

        with paths["status"].open(encoding="utf-8") as file:
            record = json.load(file)

        rows.append(record["summary"])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_name(SUMMARY_PATH.name + ".tmp")

    with temporary.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=SUMMARY_COLUMNS
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, SUMMARY_PATH)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or greater")

    captures = discover_captures()
    pipeline_hash = pipeline_signature()

    selected = (
        captures if args.limit == 0
        else captures[:args.limit]
    )

    print(f"Discovered {len(captures)} captures")
    print(f"Selected {len(selected)} captures")

    processed = 0
    skipped = 0

    try:
        for path, capture_id in selected:
            paths = output_paths(capture_id)

            if args.force:
                paths["status"].unlink(missing_ok=True)

            result = process_capture(
                path, capture_id, pipeline_hash
            )

            if result == "processed":
                processed += 1
            else:
                skipped += 1

    finally:
        completed = save_summary(captures, pipeline_hash)

        print("\nPROCESSING SUMMARY")
        print(f"Newly processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Completed captures: {completed}/{len(captures)}")
        print(f"Summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
