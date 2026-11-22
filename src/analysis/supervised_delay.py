
"""Observation delay for the same 851 flows used in ML evaluation."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "supervised_delay"

# Load previously extracted first-N features and evaluation assignments.
features = pd.read_csv(ROOT / "data/processed/all_features.csv")
splits = pd.read_csv(
    ROOT / "results/capture_splits/flow_partition_manifest.csv"
)

# Select the held-out test flows, each appearing exactly once.
test = splits[
    (splits["design"] == "outer3_inner_holdout")
    & (splits["partition"] == "test")
][["flow_id", "capture_id", "candidate_class"]].drop_duplicates()

assert len(test) == 851
assert test["flow_id"].is_unique

# Match the same flows to their four packet-prefix measurements.
data = test.merge(
    features[["flow_id", "capture_id", "n_packets", "time_to_n"]],
    on=["flow_id", "capture_id"],
    how="inner",
    validate="one_to_many",
)

assert len(data) == 851 * 4
assert (data.groupby("flow_id")["n_packets"].nunique() == 4).all()
assert not data["time_to_n"].isna().any()
assert (data["time_to_n"] >= 0).all()

# Convert observation delay from seconds to milliseconds.
data["delay_ms"] = data["time_to_n"] * 1000

def summarize(group):
    return pd.Series({
        "flows": len(group),
        "mean_ms": group["delay_ms"].mean(),
        "median_ms": group["delay_ms"].median(),
        "p90_ms": group["delay_ms"].quantile(0.90),
        "p95_ms": group["delay_ms"].quantile(0.95),
    })

overall = (
    data.groupby("n_packets")
    .apply(summarize, include_groups=False)
    .reset_index()
)

by_class = (
    data.groupby(["candidate_class", "n_packets"])
    .apply(summarize, include_groups=False)
    .reset_index()
)

OUTPUT.mkdir(parents=True, exist_ok=True)

overall.to_csv(OUTPUT / "supervised_delay_overall.csv", index=False)
by_class.to_csv(OUTPUT / "supervised_delay_by_class.csv", index=False)

# Generate research figure.
plt.figure(figsize=(7, 4.5))
plt.plot(overall["n_packets"], overall["median_ms"],
         marker="o", label="Median")
plt.plot(overall["n_packets"], overall["p95_ms"],
         marker="o", label="95th percentile")

plt.yscale("log")
plt.xticks([3, 5, 10, 20])
plt.xlabel("Observed packets (N)")
plt.ylabel("Observation delay (ms, log scale)")
plt.title("Observation Delay — 851 Supervised Flows")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT / "supervised_delay.png", dpi=300)
plt.close()

print("Supervised delay analysis completed!")
print(overall.round(3).to_string(index=False))
print(f"Results saved in: {OUTPUT}")
