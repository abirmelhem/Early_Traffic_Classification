
# Test A2–A3 — Bidirectional Flow Construction
# Verify packet filtering, bidirectional grouping, and inactivity timeouts.
# Test flow statistics and first-N packet storage using synthetic packets.

from collections import Counter

import pytest
from scapy.all import IP, TCP, UDP, ARP

from src.traffic import flows


def make_packet(
    src="192.168.1.10",
    dst="192.168.1.20",
    sport=50000,
    dport=443,
    protocol="TCP",
    timestamp=0.0,
):
    transport = (
        TCP(sport=sport, dport=dport)
        if protocol == "TCP"
        else UDP(sport=sport, dport=dport)
    )

    packet = IP(src=src, dst=dst) / transport
    packet.time = timestamp

    return packet


def collect_flows(monkeypatch, packets):
    monkeypatch.setattr(
        flows,
        "iter_packets",
        lambda path: iter(packets),
    )

    counters = Counter()

    result = list(
        flows.iter_flows(
            "synthetic.pcap",
            "test/synthetic.pcap",
            counters,
        )
    )

    return result, counters


def test_packet_filtering():
    valid = make_packet()

    non_ipv4 = ARP()
    non_tcp_udp = IP(
        src="192.168.1.10",
        dst="192.168.1.20",
        proto=1,
    )

    multicast = make_packet(dst="224.0.0.1")
    unspecified = make_packet(src="0.0.0.0")
    broadcast = make_packet(dst="255.255.255.255")
    directed = make_packet(dst="131.202.243.255")

    assert flows.packet_filter_reason(valid) is None
    assert flows.packet_filter_reason(non_ipv4) == "non_ipv4"
    assert flows.packet_filter_reason(non_tcp_udp) == "non_tcp_udp"
    assert flows.packet_filter_reason(multicast) == "multicast"
    assert (
        flows.packet_filter_reason(unspecified)
        == "unspecified_address"
    )
    assert flows.packet_filter_reason(broadcast) == "broadcast"
    assert (
        flows.packet_filter_reason(directed)
        == "directed_broadcast"
    )

    assert flows.valid_packet(valid)
    assert not flows.valid_packet(multicast)


def test_bidirectional_flow_key():
    forward = make_packet(
        src="192.168.1.10",
        dst="192.168.1.20",
        sport=50000,
        dport=443,
    )

    backward = make_packet(
        src="192.168.1.20",
        dst="192.168.1.10",
        sport=443,
        dport=50000,
    )

    assert flows.get_flow_key(forward) == flows.get_flow_key(
        backward
    )


def test_separate_connections():
    first = make_packet(
        sport=50000,
        timestamp=1.0,
    )

    second = make_packet(
        sport=50001,
        timestamp=2.0,
    )

    tcp_flows = {
        flows.get_flow_key(first),
        flows.get_flow_key(second),
    }

    udp = make_packet(
        sport=50000,
        protocol="UDP",
        timestamp=3.0,
    )

    assert len(tcp_flows) == 2
    assert flows.get_flow_key(first) != flows.get_flow_key(udp)


def test_bidirectional_flow_construction(monkeypatch):
    packets = [
        make_packet(timestamp=1.0),
        make_packet(
            src="192.168.1.20",
            dst="192.168.1.10",
            sport=443,
            dport=50000,
            timestamp=2.0,
        ),
        make_packet(timestamp=3.0),
    ]

    result, counters = collect_flows(
        monkeypatch,
        packets,
    )

    assert len(result) == 1

    flow = result[0]

    assert flow["packet_count"] == 3
    assert flow["forward_count"] == 2
    assert flow["backward_count"] == 1
    assert flow["protocol"] == "TCP"
    assert flow["first_time"] == 1.0
    assert flow["last_time"] == 3.0

    assert [
        item["direction"]
        for item in flow["prefix_packets"]
    ] == [1, -1, 1]

    assert counters["total"] == 3
    assert counters["eligible"] == 3
    assert counters["filtered"] == 0


def test_flow_timeout(monkeypatch):
    packets = [
        make_packet(timestamp=1.0),
        make_packet(timestamp=2.0),
        make_packet(
            timestamp=2.0 + flows.TIMEOUT + 1.0
        ),
    ]

    result, counters = collect_flows(
        monkeypatch,
        packets,
    )

    assert len(result) == 2
    assert sorted(
        flow["packet_count"] for flow in result
    ) == [1, 2]

    assert len({
        flow["flow_id"] for flow in result
    }) == 2

    assert counters["eligible"] == 3


def test_filtering_counters(monkeypatch):
    packets = [
        make_packet(timestamp=1.0),
        make_packet(
            dst="224.0.0.1",
            timestamp=2.0,
        ),
        make_packet(
            dst="255.255.255.255",
            timestamp=3.0,
        ),
    ]

    result, counters = collect_flows(
        monkeypatch,
        packets,
    )

    assert len(result) == 1
    assert result[0]["packet_count"] == 1

    assert counters["total"] == 3
    assert counters["eligible"] == 1
    assert counters["filtered"] == 2
    assert counters["multicast"] == 1
    assert counters["broadcast"] == 1


def test_prefix_packet_limit(monkeypatch):
    packets = [
        make_packet(timestamp=float(i))
        for i in range(flows.MAX_PREFIX + 5)
    ]

    result, _ = collect_flows(
        monkeypatch,
        packets,
    )

    assert len(result) == 1

    flow = result[0]

    assert flow["packet_count"] == flows.MAX_PREFIX + 5
    assert len(flow["prefix_packets"]) == flows.MAX_PREFIX

    assert [
        item["timestamp"]
        for item in flow["prefix_packets"]
    ] == [
        float(i) for i in range(flows.MAX_PREFIX)
    ]


def test_statistics_row(monkeypatch):
    packets = [
        make_packet(timestamp=float(i))
        for i in range(1, 11)
    ]

    result, _ = collect_flows(
        monkeypatch,
        packets,
    )

    row = flows.statistics_row(result[0])

    assert row["packet_count"] == 10
    assert row["forward_count"] == 10
    assert row["backward_count"] == 0
    assert row["duration_seconds"] == 9.0

    for n in flows.PACKET_BUDGETS:
        assert row[f"eligible_n{n}"] == int(10 >= n)


def test_flow_id_is_deterministic(monkeypatch):
    packets = [
        make_packet(timestamp=1.0),
        make_packet(timestamp=2.0),
    ]

    first, _ = collect_flows(monkeypatch, packets)
    second, _ = collect_flows(monkeypatch, packets)

    assert first[0]["flow_id"] == second[0]["flow_id"]


def test_timeout_boundary(monkeypatch):
    packets = [
        make_packet(timestamp=1.0),
        make_packet(
            timestamp=1.0 + flows.TIMEOUT
        ),
    ]

    result, _ = collect_flows(
        monkeypatch,
        packets,
    )

    assert len(result) == 1
    assert result[0]["packet_count"] == 2
