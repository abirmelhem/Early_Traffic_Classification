
# A2–A3 — Bidirectional Flow Construction
# Filter packets and group TCP/UDP packets into bidirectional flows.
# Store the first N packets and generate flow statistics.

from collections import defaultdict
import hashlib
import heapq
import ipaddress

from scapy.all import IP, TCP, UDP

from config import MAX_PREFIX, TIMEOUT, PACKET_BUDGETS
from src.traffic.parse_pcap import iter_packets


SUBNET = ipaddress.ip_network("131.202.240.0/22")

FILTER_REASONS = [
    "non_ipv4",
    "non_tcp_udp",
    "multicast",
    "unspecified_address",
    "broadcast",
    "directed_broadcast",
]

STATISTICS_COLUMNS = [
    "flow_id",
    "capture_id",
    "protocol",
    "packet_count",
    "forward_count",
    "backward_count",
    "duration_seconds",
    *[f"eligible_n{n}" for n in PACKET_BUDGETS],
]


def packet_filter_reason(packet):
    if IP not in packet:
        return "non_ipv4"

    if TCP not in packet and UDP not in packet:
        return "non_tcp_udp"

    src = ipaddress.ip_address(packet[IP].src)
    dst = ipaddress.ip_address(packet[IP].dst)

    if src.is_multicast or dst.is_multicast:
        return "multicast"

    if src.is_unspecified or dst.is_unspecified:
        return "unspecified_address"

    if dst == ipaddress.ip_address("255.255.255.255"):
        return "broadcast"

    if dst == SUBNET.broadcast_address:
        return "directed_broadcast"

    return None


def valid_packet(packet):
    return packet_filter_reason(packet) is None


def get_flow_key(packet):
    ip = packet[IP]
    transport = packet[TCP] if TCP in packet else packet[UDP]
    protocol = "TCP" if TCP in packet else "UDP"

    src = (ip.src, int(transport.sport))
    dst = (ip.dst, int(transport.dport))

    return protocol, tuple(sorted((src, dst)))


def iter_flows(path, capture_id, counters):
    active = {}
    expiration_heap = []
    sequences = defaultdict(int)
    latest_timestamp = float("-inf")

    for packet in iter_packets(path):
        counters["total"] += 1

        reason = packet_filter_reason(packet)

        if reason is not None:
            counters["filtered"] += 1
            counters[reason] += 1
            continue

        counters["eligible"] += 1

        timestamp = float(packet.time)
        latest_timestamp = max(latest_timestamp, timestamp)

        while expiration_heap:
            last_time, version, expired_key = expiration_heap[0]

            if latest_timestamp - last_time <= TIMEOUT:
                break

            heapq.heappop(expiration_heap)
            expired_flow = active.get(expired_key)

            if expired_flow is None:
                continue

            if expired_flow["version"] != version:
                continue

            yield expired_flow
            del active[expired_key]

        key = get_flow_key(packet)
        flow = active.get(key)

        if flow is not None and timestamp - flow["last_time"] > TIMEOUT:
            yield flow
            del active[key]
            flow = None

        if flow is None:
            sequences[key] += 1
            raw_id = f"{capture_id}|{key}|{sequences[key]}"
            flow_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

            transport = packet[TCP] if TCP in packet else packet[UDP]
            source = (packet[IP].src, int(transport.sport))

            flow = {
                "flow_id": flow_id,
                "capture_id": capture_id,
                "protocol": key[0],
                "first_time": timestamp,
                "last_time": timestamp,
                "packet_count": 0,
                "forward_count": 0,
                "backward_count": 0,
                "initial_source": source,
                "prefix_packets": [],
                "version": 0,
            }

            active[key] = flow

        transport = packet[TCP] if TCP in packet else packet[UDP]
        source = (packet[IP].src, int(transport.sport))
        direction = 1 if source == flow["initial_source"] else -1

        flow["packet_count"] += 1
        flow["last_time"] = max(flow["last_time"], timestamp)
        flow["version"] += 1

        if direction == 1:
            flow["forward_count"] += 1
        else:
            flow["backward_count"] += 1

        if len(flow["prefix_packets"]) < MAX_PREFIX:
            flow["prefix_packets"].append({
                "timestamp": timestamp,
                "length": len(packet),
                "direction": direction,
            })

        heapq.heappush(
            expiration_heap,
            (flow["last_time"], flow["version"], key)
        )

    for flow in active.values():
        yield flow


def statistics_row(flow):
    count = flow["packet_count"]

    return {
        "flow_id": flow["flow_id"],
        "capture_id": flow["capture_id"],
        "protocol": flow["protocol"],
        "packet_count": count,
        "forward_count": flow["forward_count"],
        "backward_count": flow["backward_count"],
        "duration_seconds": round(
            flow["last_time"] - flow["first_time"], 6
        ),
        **{
            f"eligible_n{n}": int(count >= n)
            for n in PACKET_BUDGETS
        },
    }
