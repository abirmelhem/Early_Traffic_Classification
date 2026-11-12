# Capture-derived labeling report

The manifest covers all 49 existing capture IDs. 35 captures receive unverified
candidate activities; 14 remain unresolved. All 13,314 flows and 8,286 feature
records are retained, including unresolved records. Original CSV bytes are
unchanged (SHA-256 evidence in `validation.json`). All expected first-N feature
representations exist exactly once and agree with flow capture/protocol metadata.

## Candidate support

These are capture-context weak-label counts, not verified application-flow counts.

| Candidate | Captures | N=3 | N=5 | N=10 | N=20 |
|---|---:|---:|---:|---:|---:|
| BROWSING | 0 | 0 | 0 | 0 | 0 |
| CHAT | 15 | 387 | 246 | 156 | 92 |
| STREAMING | 0 | 0 | 0 | 0 | 0 |
| MAIL | 4 | 439 | 349 | 283 | 247 |
| VOIP | 12 | 986 | 710 | 576 | 389 |
| P2P | 0 | 0 | 0 | 0 | 0 |
| FT | 4 | 324 | 227 | 197 | 123 |
| UNRESOLVED (not an activity class) | 14 | 1063 | 677 | 559 | 256 |
| Total | 49 | 3199 | 2209 | 1771 | 1107 |

CHAT is below the default descriptive N=20 count screen of 100. BROWSING,
STREAMING, and P2P have no assigned captures. Passing that count screen is not
evidence of statistical adequacy. All recording groups are unknown, so independent
train/validation/test feasibility remains blocked for every represented candidate.
There are zero verified flow labels and zero documented independent groups.

## Decisions and unresolved captures

Documented application/activity families produce CHAT, MAIL, VOIP, and FT candidates.
No blanket browsing/background classification is performed. Gmail chat remains
unresolved rather than being inferred from the official mail-protocol description.

The complete capture IDs and rule sources are in `unresolved_captures.csv`:

- Folder 01: `facebook_video1a.pcap`, `facebook_video1b.pcapng`,
  `facebook_video2a.pcap`, `facebook_video2b.pcapng`.
- Folder 02: `gmailchat1.pcapng`, `gmailchat2.pcapng`, `gmailchat3.pcapng`;
  `hangouts_video1b.pcapng`, `hangouts_video2a.pcap`, `hangouts_video2b.pcapng`;
  `netflix1.pcap`, `netflix2.pcap`, `netflix3.pcap`, `netflix4.pcap`.

## Files

- Implementation: `src/traffic/label_captures.py`.
- Tests: `tests/test_label_captures.py`.
- Rules, evidence, null conventions, and reproduction: `docs/capture_labeling.md`.
- Manifest: `results/labeling/capture_manifest.csv` (49 rows) and JSON (49 objects).
- Labeled flow output: `data/processed/capture_labeled/all_flows_capture_labeled.csv`.
- Labeled feature output: `data/processed/capture_labeled/all_features_capture_labeled.csv`.
- Counts and support flags: `results/labeling/class_by_n.csv` (32 rows).
- Unresolved list: `results/labeling/unresolved_captures.csv` (14 rows).
- Hashes and validation: `results/labeling/validation.json`.
- This report: `results/labeling/REPORT.md` (human-written; other outputs generated).

The nine labeling unit tests passed, including failed joins, duplicate/missing
representations, candidate rules, unknown patterns, metadata semantics, unchanged
original values/bytes, and deterministic fixture regeneration. No ML training or
ARFF-to-flow joins were performed. Existing preprocessing tests require additional
third-party test/runtime dependencies and are not part of this labeling test run.
