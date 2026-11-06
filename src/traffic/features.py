
# A4–A5 — Packet Prefixes and Feature Extraction
# Extract statistical features from the first 3, 5, 10, or 20 packets.
# Prepare numerical features for early traffic classification.

import statistics

from config import PACKET_BUDGETS


FEATURE_COLUMNS = [
    "flow_id",
    "capture_id",
    "n_packets",
    "protocol",
    "mean_length",
    "std_length",
    "min_length",
    "max_length",
    "total_bytes",
    "forward_ratio",
    "backward_ratio",
    "mean_interarrival",
    "std_interarrival",
    "time_to_n",
]


def extract_features(flow, n):
    packets = flow["prefix_packets"][:n]

    if len(packets) < n:
        return None

    lengths = [packet["length"] for packet in packets]
    times = [packet["timestamp"] for packet in packets]
    directions = [packet["direction"] for packet in packets]

    intervals = [
        max(0, times[i] - times[i - 1])
        for i in range(1, n)
    ]

    return {
        "flow_id": flow["flow_id"],
        "capture_id": flow["capture_id"],
        "n_packets": n,
        "protocol": flow["protocol"],
        "mean_length": statistics.mean(lengths),
        "std_length": statistics.pstdev(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_bytes": sum(lengths),
        "forward_ratio": directions.count(1) / n,
        "backward_ratio": directions.count(-1) / n,
        "mean_interarrival": statistics.mean(intervals),
        "std_interarrival": statistics.pstdev(intervals),
        "time_to_n": max(0, times[-1] - times[0]),
    }


def iter_feature_rows(flow):
    for n in PACKET_BUDGETS:
        row = extract_features(flow, n)

        if row is not None:
            yield row
