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
| `protocol` | TCP or UDP | No; recorded metadata only |
| `mean_length` | Mean packet length over the first N packets, in bytes | Yes |
| `std_length` | Population standard deviation of packet lengths, in bytes | Yes |
| `min_length` | Minimum packet length among the first N packets, in bytes | Yes |
| `max_length` | Maximum packet length among the first N packets, in bytes | Yes |
| `total_bytes` | Sum of packet lengths over the first N packets, in bytes | No; redundant with N and mean length |
| `forward_ratio` | Proportion of the first N packets traveling in the initial source direction | Yes |
| `backward_ratio` | Proportion of the first N packets traveling in the reverse direction | No; redundant with forward ratio |
| `mean_interarrival` | Mean time between consecutive observed packets, in seconds | Yes |
| `std_interarrival` | Population standard deviation of interarrival times, in seconds | Yes |
| `time_to_n` | Nonnegative first-to-Nth timestamp difference, in seconds | No; acquisition-delay outcome |

Packet length is calculated using `len(packet)` from Scapy.

Standard deviations use Python's `statistics.pstdev()` function.

Interarrival times are calculated between consecutive observed packets in capture
order, without timestamp sorting. Negative timestamp differences are clipped to
zero. `time_to_n` also clips a negative first-to-Nth difference to zero; it is not
the sum of individually clipped interarrivals for a nonmonotonic timestamp sequence.

The seven actual predictors are `mean_length`, `std_length`, `min_length`,
`max_length`, `forward_ratio`, `mean_interarrival`, and `std_interarrival`, enforced
by `src/ml/features.py`. Other recorded columns are not additional model features.

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

The inventory's `service_label` and `session_id` fields are not verified labels or
independent session assignments. The separate capture-derived labeling pipeline is
complete, but application labels and recording independence remain unverified.

Capture filenames may suggest candidate application classes, but they do not prove that every extracted flow belongs to that application.

The completed experiments explicitly use weak labels joined by `capture_id` in
`data/processed/capture_labeled/all_flows_capture_labeled.csv` and
`data/processed/capture_labeled/all_features_capture_labeled.csv`. Added metadata
includes `application`, `candidate_class`, `label_source`, `confidence`,
`verification_status`, and `recording_group`. The label manifest is
`results/labeling/capture_manifest.csv`. All 49 recording groups remain unknown;
`verification_status` is `capture_derived_unverified`. These fields are excluded
from predictors. See [capture labeling](capture_labeling.md).

## 6. Experimental Use

The research compares classification performance at N = 3, 5, 10, and 20 packets.

- Train separate models for each packet budget.
- Use the same eligible evaluation flows across all four budgets for the primary controlled comparison.
- Report flow coverage separately for each packet budget.
- Use the saved four-fold capture-disjoint training/validation/test assignments;
  independent recording-session groups have not been established.
- Preserve natural class imbalance and identify majority/minority classes from training data.
- Evaluate per-class precision, recall, F1-score, macro-F1, and confusion matrices.
- Report observation delay using `time_to_n`.

**Important:** `time_to_n` is primarily a networking latency measurement and should be excluded from the default classifier predictors to avoid mixing the observation-delay outcome with the predictive feature set.

The final classifier experiments exclude `time_to_n`. Supervised acquisition-delay
tables convert seconds to milliseconds for the fixed 851-flow cohort. Eligibility
coverage uses all 13,314 extracted flows. Acquisition delay is not inference latency.
See the [completed protocol](experiment_protocol.md) for selection and refitting.

## 7. Current Dataset Status

The initial two NonVPN folders contain 49 processed captures.

- Total extracted flows: 13,314
- Flows eligible at N=3: 3,199
- Flows eligible at N=5: 2,209
- Flows eligible at N=10: 1,771
- Flows eligible at N=20: 1,107
- Extracted feature rows: 8,286
- Dataset validation issues detected: 0

Thirty-five captures have assigned candidates and 14 are unresolved. Excluding
256 unresolved N=20 flows leaves 851 supervised flows and 3,404 repeated flow-budget
records. Their class supports are CHAT=92, MAIL=247, VOIP=389, FT=123. The saved
`results/ml_readiness/common_cohort_manifest.csv` fixes this population. These counts
describe the completed study; future expansions are separate populations.

Passing structural validation and unit tests does not establish application-label correctness, capture independence, or model generalization.
