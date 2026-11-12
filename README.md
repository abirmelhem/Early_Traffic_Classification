# Early Network Traffic Classification Using Packet Prefixes

**Advanced Computer Networks — Research Project**

## Overview

This project investigates how the number of observed packets affects early network traffic classification, with particular attention to differences in classification performance between majority and minority application classes.

Traditional flow-based classification may rely on information collected over an entire network flow. Early traffic classification instead aims to identify an application using only the first few packets, enabling earlier network monitoring and traffic management decisions.

The project evaluates four packet observation budgets: **N = 3, 5, 10, and 20**.

### Research Question

**How does the number of observed packets affect class-specific early network traffic classification, particularly for majority and minority classes?**

The analysis focuses on the trade-off between classification reliability, flow coverage, and observation delay.

## Dataset

**Dataset:** ISCXVPN2016 — VPN-nonVPN Traffic Dataset  
**Source:** https://www.unb.ca/cic/datasets/vpn.html

The study uses NonVPN PCAP/PCAPNG captures containing traffic associated with different applications and services.

The preprocessing pipeline supports multiple capture folders, configured in `config.py`.

Original captures and generated datasets are stored locally and excluded from GitHub because of their size.

Application labels and recording/session groups must be validated before supervised learning. A capture filename alone does not guarantee that every flow belongs to the named application.

## Methodology

The project follows a preprocessing and machine-learning evaluation pipeline.

### 1. Packet Processing and Flow Construction

- Read PCAP/PCAPNG files sequentially using Scapy.
- Filter packets according to the configured traffic selection rules.
- Construct bidirectional TCP/UDP flows using source/destination IP addresses, ports, and transport protocol.
- Apply a 60-second inactivity timeout to separate flows.
- Retain the first 20 packets of each flow for prefix-based feature extraction.

### 2. Feature Extraction

For each eligible flow, statistical features are extracted independently from the first N packets.

| Feature group | Extracted features |
|---|---|
| Packet lengths | Mean, standard deviation, minimum, maximum |
| Traffic volume | Total observed bytes |
| Packet direction | Forward and backward packet ratios |
| Timing | Mean and standard deviation of packet interarrival times |
| Protocol | TCP or UDP |
| Observation delay | Time required to observe the first N packets |

Features are calculated using only information available within the corresponding packet prefix.

Full-flow statistics and information from packets beyond N are excluded from early-classification predictors.

### 3. Machine-Learning Evaluation

The experimental design compares:

- **DummyClassifier:** Baseline representing a simple prediction strategy.
- **Random Forest:** Supervised classifier using first-N packet features.

Separate models are trained for each packet budget.

For the primary controlled comparison, the same eligible test flows are evaluated at N = 3, 5, 10, and 20 to isolate the effect of packet observation length.

Flow coverage is reported separately to account for connections that do not contain enough packets.

### 4. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision | Measure correctness of positive predictions for each class |
| Recall | Measure how many actual instances of each class are identified |
| F1-score | Balance precision and recall |
| Macro-F1 | Evaluate performance across classes with equal class weighting |
| Confusion matrix | Identify classification errors between applications |
| Class support | Show the number of evaluated samples per class |
| Flow coverage | Measure the proportion of flows eligible at each N |
| Observation delay | Measure time required to observe N packets |

The analysis preserves natural class imbalance and compares majority and minority classes based on training-set class frequencies.

## Project Structure

```text
CMPS342_Early_Traffic_Classification/
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── data_dictionary.md
│   └── experiment_protocol.md
│
├── results/
│   ├── dataset/
│   └── figures/
│
├── src/
│   ├── traffic/
│   │   ├── inventory.py
│   │   ├── parse_pcap.py
│   │   ├── flows.py
│   │   ├── features.py
│   │   ├── export.py
│   │   ├── validate_dataset.py
│   │   └── consolidate_dataset.py
│   │
│   └── ml/
│       ├── data.py
│       ├── splits.py
│       ├── train.py
│       ├── evaluate.py
│       ├── plots.py
│       └── experiments.py
│
└── tests/
    ├── test_inventory.py
    ├── test_flows.py
    ├── test_features.py
    └── test_splits.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/CMPS342_Early_Traffic_Classification.git
cd CMPS342_Early_Traffic_Classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the dataset

Download the required NonVPN captures from the official dataset source and extract them into the directories specified in `config.py`.

By default, the project searches under the user's Downloads directory. The `TRAFFIC_DATA_ROOT` environment variable can be used to specify a different dataset root.

## Running the Preprocessing Pipeline

From the project root:

```bash
python -m src.traffic.inventory
python -m src.traffic.export
python -m src.traffic.validate_dataset
python -m src.traffic.consolidate_dataset
```

The pipeline produces capture metadata, flow statistics, first-N feature datasets, and validation reports.

### Main Data Outputs

| Output | Description |
|---|---|
| `results/dataset_inventory.csv` | Capture metadata |
| `data/processed/all_flows.csv` | Extracted flow statistics |
| `data/processed/all_features.csv` | First-N packet features |
| `results/dataset/dataset_summary.csv` | Consolidated processing and validation statistics |
| `results/dataset/validation_issues.csv` | Structural validation issues |

Generated data files are excluded from version control and must be shared separately when needed.

## Testing

### Dataset EDA

Regenerate statistical tables, 300-dpi PNG figures, and the research report from
the separate capture-labeled CSVs, preserving all source data:

```bash
python -m src.analysis.dataset_eda
python -m unittest discover -s tests -p test_dataset_eda.py -v
```

See [the generated EDA report](results/eda/REPORT.md). Results describe unverified
candidate labels and packet-budget eligibility; they are not classifier results.

### Capture-derived candidate labels

Generate separate, explicitly unverified capture-context annotations for the
existing processed tables (standard library only; no model training):

```bash
python -m src.traffic.label_captures
python -m unittest discover -s tests -p test_label_captures.py -v
```

See [capture labeling rules and output documentation](docs/capture_labeling.md).
These candidates are not verified flow-level ground truth. Unknown recording
groups remain null; video, Netflix, Gmail chat, and unknown patterns remain
unresolved. Original processed CSVs are preserved.

Unit tests verify the correctness of dataset discovery, packet filtering, flow construction, timeout behavior, feature calculations, and packet-prefix independence.

Run preprocessing tests with:

```bash
python -m pytest tests/test_inventory.py tests/test_flows.py tests/test_features.py -v
```

Tests use small synthetic inputs and do not require downloading or processing the full dataset.

## Experimental Considerations

To support reliable and reproducible evaluation:

- Prevent train/test leakage by grouping related captures or recording sessions where possible.
- Maintain consistent evaluation samples across packet budgets for the primary comparison.
- Avoid using capture identifiers, flow identifiers, or post-observation information as predictive features.
- Distinguish capture-level application metadata from verified flow-level labels.
- Report class-specific performance rather than relying only on overall accuracy.
- Consider the effects of short flows, class imbalance, observation delay, and dataset limitations when interpreting results.

## Documentation

- **`docs/data_dictionary.md`** — Feature definitions, units, and data schema.
- **`docs/experiment_protocol.md`** — Experimental design, splitting strategy, and evaluation criteria.

## Contributors

- **Abir Melhem**
- **Aya Abdallah**

## Dataset Attribution

ISCXVPN2016 — Canadian Institute for Cybersecurity, University of New Brunswick.

https://www.unb.ca/cic/datasets/vpn.html
