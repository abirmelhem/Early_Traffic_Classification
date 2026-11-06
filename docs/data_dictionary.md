# Data Dictionary — Early Network Traffic Classification

## 1. Dataset Overview

**Dataset:** ISCXVPN2016 — NonVPN PCAP captures  
**Task:** Multiclass application/service traffic classification  
**Packet budgets:** N = 3, 5, 10, 20  
**Flow definition:** Bidirectional TCP/UDP flow identified by protocol and endpoint IP addresses/ports.

Each eligible flow produces one feature row for every supported packet budget N that does not exceed its observed packet count.

Features are computed exclusively from the first N packets of each flow.

## 2. Flow Statistics

**File:** `data/processed/all_flows.csv`

| Column | Description |
|---|---|
| `flow_id` | Deterministic identifier generated from capture ID, flow key, and flow sequence |
| `capture_id` | Identifier of the original PCAP/PCAPNG capture |
| `protocol` | Transport protocol: TCP or UDP |
| `packet_count` | Total number of packets in the extracted flow |
| `forward_count` | Number of packets traveling in the initial source direction |
| `backward_count` | Number of packets traveling in the opposite direction |
| `duration_seconds` | Time between the first and last packets of the flow, in seconds |
| `eligible_n3` | 1 if the flow contains at least 3 packets; otherwise 0 |
| `eligible_n5` | 1 if the flow contains at least 5 packets; otherwise 0 |
| `eligible_n10` | 1 if the flow contains at least 10 packets; otherwise 0 |
| `eligible_n20` | 1 if the flow contains at least 20 packets; otherwise 0 |

Flow statistics are used for dataset analysis, eligibility checks, and flow coverage reporting.

Full-flow statistics must not be used as predictors in first-N packet classification because they may contain information unavailable at the observation point.

## 3. Extracted Features

**File:** `data/processed/all_features.csv`

One row represents one flow at one packet budget.

| Column | Description | ML predictor? |
|---|---|---|
| `flow_id` | Unique flow identifier | No |
| `capture_id` | Original capture identifier | No |
| `n_packets` | Number of observed packets: 3, 5, 10, or 20 | No |
| `protocol` | TCP or UDP | Yes, after encoding |
| `mean_length` | Mean packet length over the first N packets, in bytes | Yes |
| `std_length` | Population standard deviation of packet lengths, in bytes | Yes |
| `min_length` | Minimum packet length among the first N packets, in bytes | Yes |
| `max_length` | Maximum packet length among the first N packets, in bytes | Yes |
| `total_bytes` | Sum of packet lengths over the first N packets, in bytes | Yes |
| `forward_ratio` | Proportion of the first N packets traveling in the initial source direction | Yes |
| `backward_ratio` | Proportion of the first N packets traveling in the reverse direction | Yes |
| `mean_interarrival` | Mean time between consecutive observed packets, in seconds | Yes |
| `std_interarrival` | Population standard deviation of interarrival times, in seconds | Yes |
| `time_to_n` | Elapsed time from the first to the Nth packet, in seconds | Report separately by default |

Packet length is calculated using `len(packet)` from Scapy.

Standard deviations use Python's `statistics.pstdev()` function.

Interarrival times are calculated between consecutive observed packets. Negative timestamp differences are clipped to zero.

The direction of the first observed packet is defined as forward (+1); packets in the opposite direction are backward (−1).

## 4. Packet Filtering and Flow Construction

The preprocessing pipeline:

- Accepts IPv4 packets containing TCP or UDP.
- Excludes non-IPv4 packets and IPv4 packets without TCP/UDP.
- Excludes multicast, unspecified-address, limited-broadcast, and configured directed-broadcast traffic.
- Groups packets bidirectionally using protocol and endpoint IP/port pairs.
- Uses a 60-second inactivity timeout to separate flows.
- Stores the first 20 packets of each flow for prefix feature extraction.

The current directed-broadcast exclusion uses the configured subnet `131.202.240.0/22`. Its applicability to every capture remains a methodological assumption requiring verification.

## 5. Application Labels and Session Metadata

**File:** `results/dataset_inventory.csv`

| Column | Description |
|---|---|
| `capture_id` | Unique identifier for a source capture |
| `filename` | Original PCAP/PCAPNG filename |
| `format` | Capture format |
| `size_mb` | Capture file size in MiB, rounded to two decimals |
| `service_label` | Application/service category, once verified |
| `session_id` | Independent recording/session identifier, once established |

Application labels and session identifiers have not yet been verified.

Capture filenames may suggest candidate application classes, but they do not prove that every extracted flow belongs to that application.

Verified labels must be joined to feature rows through the appropriate capture/flow mapping before supervised learning.

## 6. Experimental Use

The research compares classification performance at N = 3, 5, 10, and 20 packets.

- Train separate models for each packet budget.
- Use the same eligible evaluation flows across all four budgets for the primary controlled comparison.
- Report flow coverage separately for each packet budget.
- Use capture/session-grouped splitting where defensible to reduce data leakage.
- Preserve natural class imbalance and identify majority/minority classes from training data.
- Evaluate per-class precision, recall, F1-score, macro-F1, and confusion matrices.
- Report observation delay using `time_to_n`.

**Important:** `time_to_n` is primarily a networking latency measurement and should be excluded from the default classifier predictors to avoid mixing the observation-delay outcome with the predictive feature set.

## 7. Current Dataset Status

The initial two NonVPN folders contain 49 processed captures.

- Total extracted flows: 13,314
- Flows eligible at N=3: 3,199
- Flows eligible at N=5: 2,209
- Flows eligible at N=10: 1,771
- Flows eligible at N=20: 1,107
- Extracted feature rows: 8,286
- Dataset validation issues detected: 0

These counts will be updated after processing any additional captures.

Passing structural validation and unit tests does not establish application-label correctness, capture independence, or model generalization.