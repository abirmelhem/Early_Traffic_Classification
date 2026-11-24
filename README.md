# Early Network Traffic Classification under Class Imbalance

**Aya Abdallah and Abir Melhem**  
American University of Beirut — CMPS 342, Advanced Computer Networks

## Research overview

Early traffic classification uses a short packet prefix to identify traffic before a flow ends. This can support timely network monitoring and inform traffic-management decisions. Our research question is:

**How does packet observation length affect classification performance for majority and minority traffic classes in imbalanced network traffic?**

We compare the first **3, 5, 10, and 20 packets** using Logistic Regression, Random Forest, and XGBoost. The study contributes a controlled comparison on fixed flow identities, class-specific imbalance analysis, and separate measurements of packet acquisition delay and flow eligibility. It does not introduce a new classifier or demonstrate deployed QoS improvements.

The implementation and experiments are complete. Start with the [final results and figures](results/ml/paper_figures/README.md), [current experiment protocol](docs/experiment_protocol.md), and [reproduction guide](docs/ml_framework.md).

## Dataset and preprocessing

The study uses a processed non-VPN subset of [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html), rather than the complete dataset.

| Item | Verified count |
|---|---:|
| Processed captures | 49: 23 from archive 01 and 26 from archive 02 |
| Extracted bidirectional flows | 13,314 |
| Flow–packet-budget feature records | 8,286 |
| Captures with assigned candidate activities | 35 |
| Unresolved captures | 14 |
| Flows reaching 20 packets | 1,107 |
| Fixed supervised cohort | 851 flows from 35 captures |
| Supervised flow–budget records | 3,404 |

Capture filenames are mapped conservatively to candidate activities. The supervised task includes **CHAT, MAIL, VOIP, and FT**. Labels remain **unverified capture-derived annotations**, not independently verified application labels for individual flows. Background traffic is not silently identified or relabeled. Unresolved flows remain in descriptive population statistics and are excluded explicitly from supervised evaluation. Official ARFF rows are not joined to extracted flows. See [labeling rules](docs/capture_labeling.md).

The current extractor reads captures sequentially with Scapy, accepts filtered IPv4 TCP/UDP packets, reconstructs bidirectional endpoint/protocol flows, and applies a 60-second inactivity timeout. It retains the first 20 accepted packets in capture order, without timestamp sorting. Features for each N use only the first N retained packets. Packet length is `len(packet)` in bytes; negative interarrival differences and acquisition delays are clipped to zero. See the [data dictionary](docs/data_dictionary.md) for schemas and filtering assumptions.

The processed cohort and results are available locally. Raw captures under `data/raw/` currently comprise 23 flat filenames consistent with archive 01; a complete archive-02 raw reconstruction is not established. File presence in this workspace does not establish inclusion in a distributed repository checkout. Dataset files may need to be obtained separately.

## Experimental methodology

### Predictors and classifiers

Every model uses exactly seven predictors:

```text
mean_length, std_length, min_length, max_length,
forward_ratio, mean_interarrival, std_interarrival
```

Length statistics use bytes; interarrival statistics use seconds. Protocol, total prefix bytes, backward ratio, acquisition delay, completed-flow statistics, identifiers, labels, archives, and provenance are excluded from model inputs. Recorded columns and actual predictors are different sets.

All classifiers use median imputation within the fitting pipeline. Logistic Regression additionally uses standard scaling. The baselines use eight validation candidates per model: LR varies C; RF varies tree count, maximum depth, and minimum leaf size; XGBoost varies tree count, maximum depth, and learning rate. XGBoost runs on CPU with multiclass labels encoded internally. Exact grids and software versions are recorded in saved configurations and described in the [ML guide](docs/ml_framework.md).

### Evaluation and balancing

The fixed cohort contains CHAT=92, MAIL=247, VOIP=389, and FT=123 flows, all with representations at every N. Four outer capture-disjoint rounds use explicit training, validation, and test partitions. The same flow and capture assignments apply across budgets and models. Test partitions contain 258, 236, 185, and 172 flows; each flow is tested once per N.

Baseline candidates fit preprocessing and classifiers on training observations only. Validation macro-F1 selects parameters. A fresh pipeline then refits on training plus validation before predicting the untouched outer test partition. The implemented baseline protocol is `capture_cv_train_validation_refit_v1`.

Three strategies produce **36 final model/strategy/budget summaries**, each pooling four outer folds:

- **Baseline:** no class weighting or resampling.
- **Class weighting:** LR/RF use balanced class weights; XGBoost uses per-observation weights `M / (4 * M_class)` from the fitting pool.
- **Corrected random oversampling:** retain every original fitting observation, then append deterministic replacement draws until all classes reach the largest fitting-class count. Only the training-plus-validation refit pool is resampled; test observations are untouched.

Balancing experiments reuse each baseline's validation-selected parameters without separate retuning. They are exploratory comparisons conducted after baseline test inspection. Weighting and oversampling are separate interventions. The corrected runner records protocol `capture_cv_frozen_baseline_imbalance_v2`.

Majority/minority roles come from original training counts before balancing: MAIL and VOIP are majority; CHAT and FT are minority in every round. Final group scores average the two relevant **pooled per-class F1 scores with equal class weights**. Pooled macro-F1 is distinct from the unweighted mean of four fold macro-F1 scores. Fold sample standard deviations are descriptive, not confidence intervals. Saved outputs also include accuracy, weighted F1, per-class precision/recall/F1/support, and confusion counts. No significance tests are reported.

## Main experimental results

### Baseline pooled outer-test macro-F1

Each entry evaluates the same 851 flows.

| Packets | Logistic Regression | Random Forest | XGBoost |
|---:|---:|---:|---:|
| 3 | 0.330 | 0.791 | 0.781 |
| 5 | 0.348 | 0.851 | 0.855 |
| 10 | 0.442 | 0.802 | 0.827 |
| 20 | 0.574 | 0.744 | 0.735 |

Source: [final pooled metrics](results/ml/paper_figures/pooled_metrics.csv); full precision and all strategies are retained there.

RF and XGBoost attain their highest observed macro-F1 at five packets under all three strategies. LR macro-F1 increases through 20 packets. Majority-class mean F1 exceeds minority-class mean F1 in all 36 configurations, but minority classes do not uniformly improve with additional packets. Both balancing strategies improve LR macro-F1 at every budget; effects on the tree-based models are mixed. These are descriptive observations, not evidence that imbalance causes the differences or that five packets is universally optimal.

### Packet acquisition delay and flow eligibility

| Packets | Median acquisition delay (ms) | Eligible flows | Coverage (%) |
|---:|---:|---:|---:|
| 3 | 37.809 | 3,199 | 24.03 |
| 5 | 52.734 | 2,209 | 16.59 |
| 10 | 151.825 | 1,771 | 13.30 |
| 20 | 461.322 | 1,107 | 8.31 |

Delay uses the **fixed 851-flow supervised cohort**. Coverage uses **all 13,314 extracted flows**, including unresolved and short flows. The 1,107 eligible N=20 flows include 256 unresolved flows; evaluated classifier support remains 851 at every N. Sources: [supervised delay](results/ml/paper_figures/supervised_acquisition_delay.csv) and [coverage](results/ml/paper_figures/flow_coverage.csv).

For networking applications, packet budget selection should consider class-specific performance, classifier behavior, acquisition delay, and flow eligibility together. Acquisition delay is not computational inference latency, and eligibility is not classification accuracy. Scheduling, QoS improvements, deployment throughput, and inference costs were not experimentally measured.

## Final results, figures, and historical records

| Location | Status and contents |
|---|---|
| `results/ml/logistic_regression/`, `random_forest/`, `xgboost/` under `results/ml/` | Completed baselines; predictions, metrics, selections, fit events, configuration, and completion metadata |
| `results/ml/weighted_exploratory/` | Final LR/RF weighting sources; shared CSVs filtered by model; less execution metadata than the newer runner |
| `results/ml/imbalance_v2/logistic_regression/random_oversampling/` | Final corrected LR oversampling |
| `results/ml/imbalance_v2/random_forest/random_oversampling/` | Final corrected RF oversampling |
| `results/ml/imbalance_v2/xgboost/class_weighted/` | Final weighted XGBoost |
| `results/ml/imbalance_v2/xgboost/random_oversampling/` | Final corrected XGBoost oversampling |
| [Final paper figures and tables](results/ml/paper_figures/README.md) | Authoritative nine-combination comparison: 36 pooled summaries, per-class/fold tables, confusion matrices, delay, coverage, nine PNG/PDF figure pairs, hashes |
| [Earlier baseline analysis](results/ml/comparisons/research_analysis/REPORT.md) | Baseline-only analysis; its role summaries average folds first, unlike final pooled role summaries |
| `results/ml/oversampling_exploratory/` | Superseded bootstrap-style minority sampling that could omit originals; not conventional ROS |
| `results/ml/final_imbalance_comparison/` | Earlier comparison using historical oversampling; not the final nine-combination source |

Earlier EDA, readiness, and split reports are preserved snapshots, not current instructions to postpone all training. Their proposed Dummy/Decision Tree models, no-refit policy, seeds, and uncertainty analyses were not all used. The [current protocol](docs/experiment_protocol.md) and saved configurations govern the completed experiments. Generated reports and their checksum records are preserved unchanged.

## Repository structure

```text
config.py                     Raw discovery and extraction settings
requirements.txt              Core dependencies
requirements-ml-optional.txt   Optional XGBoost dependency
data/raw/                     Locally available captures
data/labels/                  Official auxiliary ARFF datasets
data/processed/               Consolidated, per-capture, and capture-labeled CSVs
docs/                         Current protocol, schemas, labeling, and ML guides
src/traffic/                  Inventory, extraction, validation, consolidation, labeling
src/ml/                       Loading, predictors, models, selection/refitting, balancing
src/analysis/                 EDA, readiness, split design, saved-result analysis/figures
results/dataset/              Extraction, filtering, status, and validation records
results/labeling/             Candidate-label manifest and support reports
results/eda/                  Descriptive whole-population EDA
results/ml_readiness/         Fixed cohort and earlier readiness audit
results/capture_splits/       Saved capture/flow/observation assignments
results/supervised_delay/     Delay analysis on the 851-flow cohort
results/ml/                   Experiments and final publication outputs
tests/                        Synthetic and analysis-validation tests
```

## Installation and execution

Run commands from the project root. Python 3.11 was used for the saved experiments. The examples below use Windows PowerShell and the environment interpreter directly; activation is optional.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ml-optional.txt
```

The optional file installs XGBoost. Requirements specify minimum versions, not a locked environment. Saved baseline/new-run configurations record Python 3.11.9, NumPy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, and XGBoost 3.2.0. Consult each `configuration.json` for the recorded run environment; installing current minimum-compatible packages does not guarantee bitwise reproduction.

### Inspect inputs without training or writing results

These commands validate the existing cohort, predictors, and manifests. The imbalance command additionally validates frozen baseline selections.

```powershell
.\.venv\Scripts\python.exe -B -m src.ml.run_experiments --model logistic_regression --validate-only
.\.venv\Scripts\python.exe -B -m src.ml.imbalance_experiments --model xgboost --method class_weighted --validate-only
```

### Reproduce experiments in an isolated copy

Training is not required to inspect the saved results. Baseline/new imbalance runners refuse existing experiment output directories. Use a separate project copy containing the required processed inputs and manifests, with the intended output directories absent. Preserve the supplied results; do not delete them to make a command run.

```powershell
.\.venv\Scripts\python.exe -B -m src.ml.run_experiments --model logistic_regression
.\.venv\Scripts\python.exe -B -m src.ml.run_experiments --model random_forest
.\.venv\Scripts\python.exe -B -m src.ml.run_experiments --model xgboost
```

These commands train models. Baselines must complete before balancing variants. See the [imbalance guide](docs/imbalance_experiments.md) for exact one-at-a-time commands, the historical LR/RF weighting reproduction path, and provenance differences. No training was rerun for this documentation review.

### Regenerate publication analysis without training

The figure generator reads completed experiments and recomputes analysis tables; it does not fit models. It writes new tables and plots and refuses an existing destination. Use an unused directory beneath `results/ml/paper_figures/`:

```powershell
.\.venv\Scripts\python.exe -B -m src.analysis.paper_figures --output results/ml/paper_figures/regenerated-01
```

Choose another unused name if necessary. No figures were regenerated for this documentation review.

### Earlier processing stages and tests

The [reproduction guide](docs/ml_framework.md#earlier-data-processing-stages) documents inventory, preprocessing, labeling, EDA, and split commands. Several overwrite their dedicated derived outputs; use an isolated copy when reproducing them. `config.py` searches `NonVPN-PCAPs-01` and `NonVPN-PCAPs-02` beneath `TRAFFIC_DATA_ROOT`, defaulting to Downloads; it does not automatically search the flat `data/raw/` folder.

Relevant tests can be run without research-model training:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_ml_framework.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_imbalance_experiments.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_imbalance_leakage.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_capture_splits.py -v
```

These test commands are provided for reproduction; they were not executed during this documentation-only review.

## Reproducibility and scientific limitations

Saved predictions, fixed manifests, selected parameters, configurations, versions, and hashes support inspection and reproduction from the supplied processed inputs. Final tables can be checked against predictions. This is not a claim of complete, independently verified raw-to-result reproduction.

- Labels are capture-derived and unverified; background-flow application identity is unresolved.
- Capture-disjoint partitions do not prove independent recording sessions. MAIL and FT each have only four captures and one test capture per outer round.
- The subset is restricted to four represented candidate activities in historical non-VPN captures. MAIL cohort flows occur only in archive 01 and FT only in archive 02.
- Selection of flows reaching 20 packets excludes short flows and conditions every classification result on this fixed cohort.
- Class frequency, protocol, application, archive, and capture setting are associated; observed differences do not establish imbalance as their cause.
- Balancing reuses baseline parameters and previously inspected tests. Comparisons are exploratory, not independently tuned confirmatory experiments.
- Four fold scores describe variation over these partitions; no statistical-significance or independent-session generalization claim follows.
- Oversampling adds copies rather than independent observations and can still permit memorization. Leakage checks do not eliminate overfitting.
- Historical extraction signatures do not match current source bytes in the saved readiness audit. Archive-02 raw verification and full historical preprocessing reconstruction remain unresolved.
- Dependency ranges are unpinned. Data availability, historical weighted-run metadata, and raw provenance limit complete reproduction.

See the [documentation review record](docs/documentation_review.md) for checked evidence, historical-document status, and remaining submission qualifications.

## Dataset attribution

ISCXVPN2016, Canadian Institute for Cybersecurity, University of New Brunswick. Dataset description and associated paper: [VPN-nonVPN dataset](https://www.unb.ca/cic/datasets/vpn.html).
