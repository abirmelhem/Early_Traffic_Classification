"""Reproduce descriptive tables and figures; no model training or raw extraction."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP", "P2P", "FT", "UNRESOLVED"]
BUDGETS = [3, 5, 10, 20]
BIN_NAMES = ["1", "2", "3–4", "5–9", "10–19", "20–99", "100–999", "1,000–9,999", "≥10,000"]
COLORS = dict(zip(CLASSES, ["#4477AA", "#228833", "#66CCEE", "#AA3377",
                           "#EE7733", "#CCBB44", "#332288", "#888888"]))


def load_inputs(root):
    folder = Path(root) / "data/processed/capture_labeled"
    paths = [folder / "all_flows_capture_labeled.csv", folder / "all_features_capture_labeled.csv"]
    flows, features = [pd.read_csv(p, dtype={"flow_id": str}, keep_default_na=False) for p in paths]
    for frame in (flows, features):
        required = {"flow_id", "capture_id", "protocol", "candidate_class", "label_source",
                    "verification_status", "recording_group"}
        if not required.issubset(frame.columns):
            raise ValueError("Missing required provenance or identity columns")
        frame["candidate_class"] = frame["candidate_class"].str.strip().replace("", "UNRESOLVED")
        if not frame["candidate_class"].isin(CLASSES).all():
            raise ValueError("Unexpected candidate class")
        if (frame["flow_id"].str.strip() == "").any() or (frame["capture_id"].str.strip() == "").any():
            raise ValueError("Empty identifiers")
        if not frame["protocol"].isin(["TCP", "UDP"]).all():
            raise ValueError("Unexpected transport protocol")
        if not frame["verification_status"].isin(["unresolved", "capture_derived_unverified"]).all():
            raise ValueError("Unexpected verification status; review evidence before analysis")
        frame["archive"] = frame["capture_id"].str.split("/").str[0]
    if flows["flow_id"].duplicated().any():
        raise ValueError("Flow table contains duplicate flow IDs")
    if features.duplicated(["flow_id", "n_packets"]).any():
        raise ValueError("Duplicate feature representation")
    counts = pd.to_numeric(flows["packet_count"], errors="raise")
    if not np.isfinite(counts).all() or (counts < 1).any() or (counts % 1 != 0).any():
        raise ValueError("Invalid packet counts")
    flows["packet_count"] = counts.astype(int)
    if flows.empty:
        raise ValueError("Empty flow table")
    expected = {(r.flow_id, n) for r in flows.itertuples() for n in BUDGETS if r.packet_count >= n}
    actual = set(zip(features.flow_id, features.n_packets))
    if expected != actual:
        raise ValueError("Missing, unknown, or ineligible feature representation")
    check = features.merge(flows, on="flow_id", suffixes=("_feature", "_flow"), validate="many_to_one")
    for col in ["capture_id", "candidate_class", "protocol", "verification_status", "label_source", "recording_group"]:
        if not check[f"{col}_feature"].eq(check[f"{col}_flow"]).all():
            raise ValueError(f"Feature/flow metadata mismatch: {col}")
    for n in BUDGETS:
        if not pd.to_numeric(flows[f"eligible_n{n}"], errors="raise").eq(counts >= n).all():
            raise ValueError("Invalid eligibility flags")
    for col in ["time_to_n", "mean_length"]:
        values = pd.to_numeric(features[col], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {col}")
        features[col] = values
    flows["length_bin"] = pd.cut(counts, [0, 1, 2, 4, 9, 19, 99, 999, 9999, np.inf], labels=BIN_NAMES)
    return paths, flows, features


def distribution(frame, index, columns=None):
    if columns:
        return pd.crosstab(frame[index], frame[columns]).reindex(columns=["TCP", "UDP"], fill_value=0)
    return frame.groupby(index).size()


def summarize(flows, features):
    total = len(flows)
    classes = flows.groupby("candidate_class").agg(
        unique_flows=("flow_id", "size"), distinct_captures=("capture_id", "nunique"))
    classes = classes.reindex(CLASSES, fill_value=0)
    classes["dataset_pct"] = 100 * classes.unique_flows / total
    labeled_total = int(classes.drop("UNRESOLVED").unique_flows.sum())
    classes["resolved_candidate_pct"] = 100 * classes.unique_flows / labeled_total if labeled_total else np.nan
    classes.loc["UNRESOLVED", "resolved_candidate_pct"] = np.nan
    eligibility = []
    for cls in CLASSES:
        subset = flows[flows.candidate_class == cls]
        for n in BUDGETS:
            eligible = subset[subset.packet_count >= n]
            denominator = int((flows.packet_count >= n).sum())
            eligibility.append(dict(candidate_class=cls, n_packets=n, total_flows=len(subset),
                                    eligible_flows=len(eligible),
                                    coverage_pct=100 * len(eligible) / len(subset) if len(subset) else np.nan,
                                    eligible_cohort_pct=100 * len(eligible) / denominator if denominator else np.nan))
    lengths = pd.crosstab(flows.candidate_class, flows.length_bin, dropna=False).reindex(index=CLASSES, columns=BIN_NAMES, fill_value=0)
    protocol = distribution(flows, "candidate_class", "protocol").reindex(CLASSES, fill_value=0)
    protocol_pct = protocol.div(protocol.sum(axis=1).replace(0, np.nan), axis=0) * 100
    protocol_detail = []
    for proto, sub in flows.groupby("protocol"):
        for n in BUDGETS:
            protocol_detail.append(dict(protocol=proto, n_packets=n, total_flows=len(sub),
                                        eligible_flows=int((sub.packet_count >= n).sum()),
                                        coverage_pct=100 * (sub.packet_count >= n).mean()))
    captures = flows.groupby(["capture_id", "candidate_class", "archive"]).agg(
        unique_flows=("flow_id", "size"), short_flows=("packet_count", lambda s: int((s < 3).sum())),
        n20_flows=("packet_count", lambda s: int((s >= 20).sum()))).reset_index()
    captures["dataset_pct"] = 100 * captures.unique_flows / total
    captures = captures.sort_values(["unique_flows", "capture_id"], ascending=[False, True])
    delay = features.assign(time_to_n_ms=features.time_to_n * 1000)
    def delay_stats(frame, keys):
        return frame.groupby(keys).time_to_n_ms.agg(
            count="size", median_ms="median", mean_ms="mean",
            p90_ms=lambda s: s.quantile(.9), p95_ms=lambda s: s.quantile(.95),
            zero_count=lambda s: int((s == 0).sum())).reset_index()
    common_ids = set(flows.loc[flows.packet_count >= 20, "flow_id"])
    archives = []
    for archive, sub in flows.groupby("archive"):
        row = dict(archive=archive, unique_flows=len(sub), captures=sub.capture_id.nunique(),
                   median_packets=sub.packet_count.median(), p95_packets=sub.packet_count.quantile(.95),
                   short_flows=int((sub.packet_count < 3).sum()))
        for n in BUDGETS:
            row[f"n{n}_flows"] = int((sub.packet_count >= n).sum())
            row[f"n{n}_coverage_pct"] = 100 * (sub.packet_count >= n).mean()
        archives.append(row)
    packets = flows.groupby("candidate_class").packet_count.agg(
        count="size", median="median", mean="mean", maximum="max",
        p90=lambda s: s.quantile(.9), p95=lambda s: s.quantile(.95), p99=lambda s: s.quantile(.99))
    loss = pd.DataFrame(eligibility).pivot(index="candidate_class", columns="n_packets", values="eligible_flows")
    loss = loss[[3, 20]].rename(columns={3: "n3_flows", 20: "n20_flows"})
    loss["lost_n3_to_n20"] = loss.n3_flows - loss.n20_flows
    loss["loss_pct_of_n3"] = 100 * loss.lost_n3_to_n20 / loss.n3_flows.replace(0, np.nan)
    overall_protocol = flows.groupby("protocol").size().rename("unique_flows").reset_index()
    overall_protocol["dataset_pct"] = 100 * overall_protocol.unique_flows / total
    return {
        "class_distribution": classes.reset_index(),
        "class_by_n": pd.DataFrame(eligibility),
        "eligibility_loss": loss.reset_index(),
        "flow_length_bins": lengths.rename_axis(columns="packet_count_bin").reset_index(),
        "flow_length_statistics": packets.reset_index(),
        "protocol_by_class": protocol.reset_index(),
        "protocol_overall": overall_protocol,
        "protocol_percent_by_class": protocol_pct.reset_index(),
        "protocol_eligibility": pd.DataFrame(protocol_detail),
        "short_flows_by_class_protocol": flows.assign(short=flows.packet_count < 3).groupby(
            ["candidate_class", "protocol"]).agg(total_flows=("flow_id", "size"), short_flows=("short", "sum")).reset_index(),
        "capture_distribution": captures,
        "delay_by_class_n": delay_stats(delay, ["candidate_class", "n_packets"]),
        "delay_by_n": delay_stats(delay, ["n_packets"]),
        "delay_common_n20_cohort": delay_stats(delay[delay.flow_id.isin(common_ids)], ["n_packets"]),
        "archive_summary": pd.DataFrame(archives),
        "archive_class_distribution": pd.crosstab(flows.archive, flows.candidate_class).reindex(columns=CLASSES, fill_value=0).reset_index(),
        "archive_flow_length_bins": pd.crosstab(flows.archive, flows.length_bin, dropna=False).reindex(columns=BIN_NAMES, fill_value=0).reset_index(),
        "prefix_mean_packet_length": features.groupby(["archive", "n_packets"]).mean_length.agg(
            count="size", median_bytes="median", mean_bytes="mean", p95_bytes=lambda s: s.quantile(.95)).reset_index(),
    }


def plot_all(tables, flows, features, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.dpi": 300})
    outputs = []
    def finish(fig, name):
        fig.text(.01, .005, "Capture-derived, unverified candidates • descriptive eligibility, not accuracy", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.savefig(folder / f"{name}.png", facecolor="white")
        plt.close(fig)
        outputs.append(name)
    classes = tables["class_distribution"].set_index("candidate_class")
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(CLASSES, classes.unique_flows, color=[COLORS[c] for c in CLASSES])
    ax.bar_label(bars, labels=[f"{int(n):,}\n{p:.1f}%" for n, p in zip(classes.unique_flows, classes.dataset_pct)], padding=3, fontsize=9)
    ax.set(ylim=(0, classes.unique_flows.max() * 1.22), ylabel="Unique flows", title=f"Full dataset: candidate distribution (n={len(flows):,} flows)")
    finish(fig, "class_distribution")
    eligible = tables["class_by_n"].pivot(index="candidate_class", columns="n_packets", values="eligible_flows").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(12, 5))
    eligible.plot.bar(ax=ax, color=["#88CCEE", "#44AA99", "#332288", "#CC6677"], rot=0)
    ax.set(ylabel="Eligible unique flows", xlabel="Candidate class", title="Eligibility by packet budget (one count per flow per N)")
    ax.legend(title="Packet budget N")
    finish(fig, "class_by_n")
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in CLASSES:
        sub = tables["class_by_n"].query("candidate_class == @cls")
        if sub.total_flows.iloc[0]:
            ax.plot(sub.n_packets, sub.coverage_pct, marker="o", color=COLORS[cls], label=f"{cls} (n={int(sub.total_flows.iloc[0]):,})")
    ax.set(xticks=BUDGETS, ylim=(0, 100), xlabel="Observed packet budget N", ylabel="Eligible / all flows within candidate (%)", title="Within-candidate packet-budget coverage")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    finish(fig, "packet_budget_coverage")
    lengths = tables["flow_length_bins"].set_index("candidate_class")
    fig, ax = plt.subplots(figsize=(11, 5))
    totals = lengths.sum(axis=0)
    bars = ax.bar(BIN_NAMES, totals, color=["#CC6677", "#CC6677"] + ["#4477AA"] * 7)
    ax.bar_label(bars, labels=[f"{v:,}" for v in totals], padding=3)
    ax.set(yscale="log", ylim=(1, max(totals) * 3), xlabel="Packets per flow (bins)", ylabel="Unique flows (log scale)", title="Flow-length distribution: one- and two-packet flows highlighted")
    finish(fig, "flow_length_distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    lengths.div(lengths.sum(axis=1).replace(0, np.nan), axis=0).mul(100).plot.bar(stacked=True, ax=ax, colormap="viridis", rot=0)
    ax.set(ylabel="Within-candidate flows (%)", xlabel="Candidate class", title="Flow-length composition by candidate (empty candidates have no data)")
    ax.legend(title="Packets", bbox_to_anchor=(1.01, 1), fontsize=8)
    finish(fig, "flow_length_by_class")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    protocols = tables["protocol_by_class"].set_index("candidate_class")
    protocols.plot.bar(stacked=True, ax=axes[0], color=["#4477AA", "#EE7733"], rot=45)
    tables["protocol_percent_by_class"].set_index("candidate_class").plot.bar(stacked=True, ax=axes[1], color=["#4477AA", "#EE7733"], rot=45)
    axes[0].set(ylabel="Unique flows", title="TCP/UDP counts"); axes[1].set(ylabel="Within-candidate flows (%)", title="TCP/UDP proportions", ylim=(0, 100))
    finish(fig, "protocol_by_class")
    fig, ax = plt.subplots(figsize=(12, 12))
    cap = tables["capture_distribution"].iloc[::-1]
    ax.barh(cap.capture_id, cap.unique_flows, color=[COLORS[c] for c in cap.candidate_class])
    ax.tick_params(axis="y", labelsize=7)
    ax.set(xlabel="Unique flows", title=f"All {len(cap)} captures (capture identity does not establish independence)")
    finish(fig, "capture_distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(CLASSES, classes.distinct_captures, color=[COLORS[c] for c in CLASSES])
    ax.bar_label(bars, padding=3); ax.set(ylabel="Distinct capture files", title="Capture support by candidate; independent recording groups unknown")
    finish(fig, "captures_per_class")
    # ECDF uses a symmetric log axis to retain exact-zero observations.
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in BUDGETS:
        values = np.sort(features.loc[features.n_packets == n, "time_to_n"].to_numpy() * 1000)
        if len(values): ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"N={n}, n={len(values):,}")
    ax.set(xscale="symlog", xlabel="Observation delay (ms; symlog, linear near zero)", ylabel="Cumulative proportion", title="Time to N: N-specific eligible cohorts")
    ax.legend(); ax.grid(alpha=.2)
    finish(fig, "observation_delay_ecdf")
    fig, ax = plt.subplots(figsize=(11, 5))
    for cls, sub in tables["delay_by_class_n"].groupby("candidate_class"):
        ax.plot(sub.n_packets, sub.median_ms, marker="o", color=COLORS[cls], label=cls)
    ax.set(yscale="symlog", xticks=BUDGETS, xlabel="Packet budget N", ylabel="Median observation delay (ms; symlog)", title="Median time to N by candidate (N-specific cohorts)")
    ax.legend(); ax.grid(alpha=.2)
    finish(fig, "observation_delay_by_class")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    tables["archive_class_distribution"].set_index("archive").plot.bar(stacked=True, ax=axes[0], color=[COLORS[c] for c in CLASSES], rot=0)
    archives = tables["archive_summary"].set_index("archive")
    for archive, row in archives.iterrows():
        axes[1].plot(BUDGETS, [row[f"n{n}_coverage_pct"] for n in BUDGETS], marker="o", label=archive)
    axes[0].set(ylabel="Unique flows", xlabel="Archive", title="Processed archive composition")
    axes[0].legend(fontsize=7); axes[1].set(xticks=BUDGETS, ylim=(0, 100), ylabel="Within-archive coverage (%)", xlabel="N", title="Archive eligibility")
    axes[1].legend()
    finish(fig, "archive_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hist = tables["archive_flow_length_bins"].set_index("archive")
    hist.div(hist.sum(axis=1), axis=0).T.mul(100).plot.bar(ax=axes[0], rot=45)
    for archive, sub in tables["prefix_mean_packet_length"].groupby("archive"):
        axes[1].plot(sub.n_packets, sub.median_bytes, marker="o", label=archive)
    axes[0].set(xlabel="Packets per flow", ylabel="Within-archive flows (%)", title="Archive flow-length distribution")
    axes[1].set(xticks=BUDGETS, xlabel="N", ylabel="Median prefix mean packet size (bytes)", title="Packet-size aggregates (eligible cohorts)")
    axes[1].legend()
    finish(fig, "archive_length_comparison")
    return outputs


def markdown(frame):
    def fmt(value):
        if pd.isna(value): return "N/A"
        if isinstance(value, (float, np.floating)): return f"{value:,.3f}"
        return str(value)
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"] + ["| " + " | ".join(r) + " |" for r in rows])


def write_report(root, output, tables, flows, features, plots):
    cls = tables["class_distribution"]
    positive = cls[(cls.candidate_class != "UNRESOLVED") & (cls.unique_flows > 0)]
    largest = positive.loc[positive.unique_flows.idxmax()] if len(positive) else None
    smallest = positive.loc[positive.unique_flows.idxmin()] if len(positive) else None
    short = flows[flows.packet_count < 3]
    n20 = flows[flows.packet_count >= 20]
    top = tables["capture_distribution"].iloc[0]
    chat_n20 = int(tables["class_by_n"].query("candidate_class == 'CHAT' and n_packets == 20").eligible_flows.iloc[0])
    absent = cls.loc[cls.unique_flows == 0, "candidate_class"].tolist()
    unknown_groups = int(flows.loc[flows.recording_group.eq(""), "capture_id"].nunique())
    losses = tables["eligibility_loss"].dropna(subset=["loss_pct_of_n3"])
    greatest_loss = losses.loc[losses.loss_pct_of_n3.idxmax()] if len(losses) else None
    n20_shares = tables["class_by_n"].query("n_packets == 20")[["candidate_class", "eligible_flows", "eligible_cohort_pct"]]
    raw02 = list((root / "data/raw").rglob("*")) if (root / "data/raw").exists() else []
    explicit02 = any(p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"} and "NonVPN-PCAPs-02" in p.parts for p in raw02)
    imbalance = (f"Among assigned candidates, {largest.candidate_class} is largest ({int(largest.unique_flows):,}) "
                 f"and {smallest.candidate_class} smallest ({int(smallest.unique_flows):,}); ratio "
                 f"{largest.unique_flows / smallest.unique_flows:.2f}:1." if largest is not None else "No assigned candidate classes.")
    content = f"""# Dataset EDA: first-N packet research

## Scope and interpretation

{len(flows):,} unique flows; {len(features):,} feature representations at N=3,5,10,20.
Feature rows are repeated representations, not independent flows. Labels are
capture-derived and unverified. UNRESOLVED is retained separately and is not an
eighth official activity class. This report describes eligibility, not accuracy.

## Provisional imbalance

{imbalance}
These are full-dataset descriptive counts, not final training-set majority/minority
definitions. Missing classes cannot be treated as minority classes with observations.

{markdown(cls)}

## Eligibility and changing cohorts

{markdown(tables['class_by_n'])}

The N=20 cohort contains {len(n20):,} flows ({100 * len(n20) / len(flows):.3f}% of all flows).
All have all four representations. Comparing each N's full eligible population
changes both class composition and flow identity. Primary classifier comparisons
must instead use a fixed common cohort and fixed group-safe partitions.

Loss between N=3 and N=20 (relative to the N=3 eligible population):

{markdown(tables['eligibility_loss'])}

Among populated categories, the greatest proportional N=3-to-N=20 attrition is
{greatest_loss.candidate_class if greatest_loss is not None else 'N/A'}
({greatest_loss.loss_pct_of_n3 if greatest_loss is not None else float('nan'):.2f}%).
Class shares within the final eligible cohort are:

{markdown(n20_shares)}

These composition shifts can change which candidates appear descriptively dominant.
They must be distinguished from differences in classifier learning difficulty.

## Short flows, protocols, and capture concentration

{len(short):,} flows contain fewer than three packets; {(short.protocol == 'UDP').sum():,}
of these are UDP. There are {(flows.packet_count == 2).sum():,} two-packet flows.
N=20 has {(n20.protocol == 'TCP').sum():,} TCP and {(n20.protocol == 'UDP').sum():,} UDP flows.
This association supports short UDP conversations as a contributor to low coverage;
it does not establish application identity, packet-loss causes, or parser correctness.

The largest capture, `{top.capture_id}`, contributes {int(top.unique_flows):,} flows
({top.dataset_pct:.2f}% of the dataset). Capture concentration reduces effective
diversity; separate capture files are not proven independent recording sessions.

{markdown(tables['capture_distribution'].head(10))}

{markdown(tables['protocol_eligibility'])}

{markdown(tables['protocol_overall'])}

## Observation delay

All delay statistics use milliseconds. Quantiles use pandas' linear interpolation.
Exact zeros are retained. ECDF/median figures use symlog axes to expose skew and
retain zeros. Aggregate-prefix observation delay is not inference or processing latency.

N-specific eligible cohorts:

{markdown(tables['delay_by_n'])}

Fixed common N=20 cohort (same identities at every N):

{markdown(tables['delay_common_n20_cohort'])}

Class-specific distributions and sample counts are in `tables/delay_by_class_n.csv`.
Changes across N-specific cohorts must not be attributed solely to packet budget.

## Archives

{markdown(tables['archive_summary'])}

Archive 02 raw PCAPs are not yet locally available for verification according to
the project context. Explicit archive-02 raw paths detected: {explicit02}.
Flat filenames do not independently establish archive provenance; no raw packets
were parsed here. Archive 02 statistics are descriptions of existing processed data.
Packet-count bins and prefix mean packet sizes are reported separately: a packet
size aggregate is not a per-packet size distribution or a full-flow mean size.

## Preliminary experiment support and limitations

{markdown(tables['class_by_n'].query('n_packets == 20')[['candidate_class','eligible_flows','coverage_pct']])}

Counts support implementation of preliminary weak-label baselines for represented
candidates. They do not establish adequate independent test support. CHAT has
{chat_n20} N=20 candidates. Candidates with no flows: {', '.join(absent) or 'none'}.
{unknown_groups} captures have unknown recording groups in the labeled dataset.
Resolve label provenance, background traffic, and independent recording groups
before claiming class-specific generalization. Validate archive 02 short flows and
historical preprocessing provenance. Neither low coverage nor minority status
implies poor accuracy or a need for more packets; classifiers have not been evaluated.

## Figures and tables

"""
    for name in plots:
        content += f"\n### {name.replace('_', ' ').title()}\n\n![{name}](plots/{name}.png)\n"
    content += "\n## Reproduction\n\n```powershell\npython -m src.analysis.dataset_eda\npython -m unittest discover -s tests -p test_dataset_eda.py -v\n```\n\n"
    content += "All statistical tables are under `tables/`; `validation.json` records input hashes, versions, and validation. Source datasets and labeling outputs are never overwritten.\n"
    (output / "REPORT.md").write_text(content, encoding="utf-8")


def run(root=ROOT):
    root = Path(root)
    sources = [root / "data/processed/capture_labeled" / f"all_{kind}_capture_labeled.csv" for kind in ("flows", "features")]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    _, flows, features = load_inputs(root)
    tables = summarize(flows, features)
    output = root / "results/eda"
    for folder in (output / "tables", output / "plots"): folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items(): table.to_csv(output / "tables" / f"{name}.csv", index=False, lineterminator="\n", float_format="%.9g")
    plots = plot_all(tables, flows, features, output / "plots")
    write_report(root, output, tables, flows, features, plots)
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "Inputs changed"
    validation = dict(unique_flows=len(flows), feature_representations=len(features),
                      class_total=int(tables["class_distribution"].unique_flows.sum()),
                      eligible_counts={str(n): int((flows.packet_count >= n).sum()) for n in BUDGETS},
                      input_sha256=hashes, sources_unchanged=True, alignment_valid=True,
                      missing_labels_normalized_to_unresolved=True,
                      versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
                      pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      plots=plots, tables=list(tables))
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
