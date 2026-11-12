# Capture-derived candidate labels

This pipeline retains the first-N packet study at N = 3, 5, 10, 20. It annotates
existing records without modifying the original CSVs, extracting packets, matching
ARFF rows, identifying background flows, or training models.

## Evidence and interpretation

The authoritative taxonomy is the [UNB ISCXVPN2016 description](https://www.unb.ca/cic/datasets/vpn.html)
and its linked paper: Draper-Gil et al., *Characterization of Encrypted and VPN
Traffic Using Time-Related Features*, ICISSP 2016, pp. 407–414.
UNB lists AIM/ICQ/Facebook/Hangouts chat, Facebook/Hangouts voice calls, email
protocols, and FTPS transfers. It also explicitly notes browsing flows within
Hangouts call captures. Thus a capture activity is not ground truth for every flow.

`candidate_class` is a weak capture-context annotation, never a verified target.
There is intentionally no column named `label` or `ground_truth`.
`verification_status=capture_derived_unverified` and
`confidence=capture_candidate_only` are categorical statements, not probabilities.
No background flow is identified, removed, or declared to belong to the candidate.
Any model later trained on these candidates must disclose capture-level weak
supervision and potential contamination. All metadata must be excluded from predictors.

## Explicit filename rules (version 1)

Rules use case-insensitive full matches on the filename stem, accept only PCAP or
PCAPNG extensions, and do not use directory names as activity evidence.
The exact regexes live in `src/traffic/label_captures.py: RULES`.

| Pattern family | Application | Candidate |
|---|---|---|
| AIM chat, ICQ chat | AIM / ICQ | CHAT |
| Facebook chat | Facebook | CHAT |
| Hangout/Hangouts chat | Hangouts | CHAT |
| email + recording number/suffix | Email client (unspecified) | MAIL |
| Facebook audio | Facebook | VOIP |
| Hangouts audio | Hangouts | VOIP |
| ftps_up / ftps_down | FTPS | FT |
| Facebook/Hangouts video | Named application | UNRESOLVED |
| Netflix | Netflix | UNRESOLVED |
| Gmail chat | Gmail chat | UNRESOLVED |
| Any other pattern | Unknown | UNRESOLVED |

Gmail chat is deliberately unresolved: the official email description establishes
mail protocols, not a mapping of these specific chat captures. The word Gmail must
not automatically become MAIL. Video is not automatically VOIP or STREAMING, and
Netflix is not in the official streaming application list inspected for this rule set.
There are no inferred BROWSING, STREAMING, or P2P captures in this subset.

`recording_group` is null in JSON and an empty cell in CSV for every capture.
Neither numeric suffixes, a/b suffixes, nor capture IDs establish independence.
Zero documented groups means unknown, not zero actual recordings. No synthetic
groups or train/test partitions are created.

## Reproduce

From the repository root, with Python 3.11 (standard library only):

```powershell
python -m src.traffic.label_captures
python -m unittest discover -s tests -p test_label_captures.py -v
```

The default input root is derived from the module location. `--root PATH` supports
isolated fixtures. `--min-n20 100` sets a descriptive count-warning threshold.
100 is an explicit engineering screen, not a power calculation or a claim of
adequate class performance. Unknown recording groups block independent split
assessment regardless of flow count. Missing official classes appear with zero
counts; UNRESOLVED is reported separately and is not an eighth activity class.

Generated files:

- `results/labeling/capture_manifest.csv` and `.json`: all inventory capture IDs,
  application, candidate class, source/rule, confidence, verification, and group.
- `data/processed/capture_labeled/all_flows_capture_labeled.csv`.
- `data/processed/capture_labeled/all_features_capture_labeled.csv`.
- `results/labeling/class_by_n.csv`: all seven classes plus unresolved, eligibility,
  capture support, unknown groups, and screening flags at every N.
- `results/labeling/unresolved_captures.csv`: explicit unresolved cases.
- `results/labeling/validation.json`: original input hashes, pipeline hash, counts,
  threshold, and validation outcomes. It omits timestamps for deterministic reruns.

All records are retained, including unresolved ones; no curated training set is
silently selected. Joining on `capture_id` appends metadata while preserving every
original column value and row order. Before writing, the pipeline rejects duplicate
flow/manifest IDs, missing or extra feature keys, inconsistent metadata/eligibility,
and inventory/flow capture-set differences. Original input bytes are hashed before
and after execution. Regeneration overwrites only these dedicated derived outputs;
the generated manifest is not a manually curated override file. Future authoritative
verification needs an explicit reviewed override workflow, not edits silently lost
on regeneration.
