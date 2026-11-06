# Test A4–A5 — First-N Feature Extraction
# Verify feature calculations and eligibility for packet budgets 3, 5, 10, and 20.
# Ensure future packets cannot affect features extracted from an earlier prefix.

import math
import statistics

import pytest

from src.traffic import features


def make_flow(count=20):
    return {
        "flow_id": "test_flow_001",
        "capture_id": "test/synthetic.pcap",
        "protocol": "TCP",
        "prefix_packets": [
            {
                "timestamp": float(i),
                "length": 100 + 10 * i,
                "direction": 1 if i % 2 == 0 else -1,
            }
            for i in range(count)
        ],
    }


def test_feature_calculations():
    flow = make_flow()

    result = features.extract_features(flow, 3)

    assert result is not None
    assert result["flow_id"] == "test_flow_001"
    assert result["capture_id"] == "test/synthetic.pcap"
    assert result["protocol"] == "TCP"
    assert result["n_packets"] == 3

    assert result["mean_length"] == pytest.approx(110.0)
    assert result["std_length"] == pytest.approx(
        statistics.pstdev([100, 110, 120])
    )
    assert result["min_length"] == 100
    assert result["max_length"] == 120
    assert result["total_bytes"] == 330

    assert result["forward_ratio"] == pytest.approx(2 / 3)
    assert result["backward_ratio"] == pytest.approx(1 / 3)

    assert result["mean_interarrival"] == pytest.approx(1.0)
    assert result["std_interarrival"] == pytest.approx(0.0)
    assert result["time_to_n"] == pytest.approx(2.0)


def test_all_packet_budgets():
    flow = make_flow(20)

    rows = list(features.iter_feature_rows(flow))

    assert len(rows) == 4
    assert [row["n_packets"] for row in rows] == [3, 5, 10, 20]

    for row in rows:
        assert row["total_bytes"] == sum(
            100 + 10 * i
            for i in range(row["n_packets"])
        )


def test_insufficient_packets():
    flow = make_flow(2)

    assert features.extract_features(flow, 3) is None
    assert list(features.iter_feature_rows(flow)) == []


def test_partial_packet_budgets():
    flow = make_flow(7)

    rows = list(features.iter_feature_rows(flow))

    assert [row["n_packets"] for row in rows] == [3, 5]


def test_no_future_packet_leakage():
    original = make_flow(20)
    modified = make_flow(20)

    for packet in modified["prefix_packets"][3:]:
        packet["length"] = 999999
        packet["timestamp"] += 10000
        packet["direction"] *= -1

    original_n3 = features.extract_features(original, 3)
    modified_n3 = features.extract_features(modified, 3)

    assert original_n3 == modified_n3

    original_n20 = features.extract_features(original, 20)
    modified_n20 = features.extract_features(modified, 20)

    assert original_n20 != modified_n20


def test_direction_ratios():
    flow = make_flow(5)

    result = features.extract_features(flow, 5)

    assert result["forward_ratio"] == pytest.approx(3 / 5)
    assert result["backward_ratio"] == pytest.approx(2 / 5)

    assert (
        result["forward_ratio"] + result["backward_ratio"]
        == pytest.approx(1.0)
    )


def test_interarrival_times():
    flow = make_flow(3)

    flow["prefix_packets"][0]["timestamp"] = 1.0
    flow["prefix_packets"][1]["timestamp"] = 2.0
    flow["prefix_packets"][2]["timestamp"] = 5.0

    result = features.extract_features(flow, 3)

    assert result["mean_interarrival"] == pytest.approx(2.0)
    assert result["std_interarrival"] == pytest.approx(1.0)
    assert result["time_to_n"] == pytest.approx(4.0)


def test_negative_timestamp_difference():
    flow = make_flow(3)

    flow["prefix_packets"][0]["timestamp"] = 5.0
    flow["prefix_packets"][1]["timestamp"] = 3.0
    flow["prefix_packets"][2]["timestamp"] = 7.0

    result = features.extract_features(flow, 3)

    assert result["mean_interarrival"] == pytest.approx(2.0)
    assert result["std_interarrival"] == pytest.approx(2.0)
    assert result["time_to_n"] == pytest.approx(2.0)


def test_feature_columns():
    flow = make_flow(3)

    result = features.extract_features(flow, 3)

    assert set(result) == set(features.FEATURE_COLUMNS)


def test_prefix_independence_from_full_flow_count():
    first = make_flow(20)
    second = make_flow(20)

    first["packet_count"] = 20
    second["packet_count"] = 1000

    assert features.extract_features(first, 5) == (
        features.extract_features(second, 5)
    )
