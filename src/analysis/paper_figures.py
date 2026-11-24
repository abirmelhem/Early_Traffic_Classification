"""Publication figures from saved predictions only. No model imports or training."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import sklearn

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("logistic_regression", "random_forest", "xgboost")
NAMES = dict(zip(MODELS, ("Logistic Regression", "Random Forest", "XGBoost")))
METHODS = ("Baseline", "Class weighting", "Corrected ROS")
CLASSES = ("CHAT", "MAIL", "VOIP", "FT")
BUDGETS = (3, 5, 10, 20)
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73")))
MARKERS = dict(zip(METHODS, ("o", "s", "^")))
STYLES = dict(zip(METHODS, ("-", "--", "-.")))
FIGURES = {
    "fig01_baseline_macro_f1": "Pooled baseline macro-F1 versus packet budget; three models.",
    "fig02_methods_macro_f1": "Pooled macro-F1 by training strategy, one panel per model.",
    "fig03_majority_minority_f1": "Mean pooled class F1: MAIL/VOIP majority and CHAT/FT minority; equal class weights.",
    **{f"fig04_{m}_per_class_f1": f"Pooled per-class F1 for {NAMES[m]}, all three strategies." for m in MODELS},
    "fig05_baseline_confusions": "Baseline RF/XGBoost at N=5 and N=20; counts and true-class row percentages.",
    "fig06_fold_macro_f1": "Four outer-fold macro-F1 trajectories for every model/strategy; descriptive variation, not confidence intervals.",
    "fig07_acquisition_delay_coverage": "Acquisition delay for 851 fixed supervised flows and eligibility coverage of all 13,314 extracted flows.",
}


def sources():
    paths = {}
    for model in MODELS:
        paths[(model, "Baseline")] = f"results/ml/{model}"
        paths[(model, "Class weighting")] = ("results/ml/weighted_exploratory" if model != "xgboost"
                                            else "results/ml/imbalance_v2/xgboost/class_weighted")
        paths[(model, "Corrected ROS")] = f"results/ml/imbalance_v2/{model}/random_oversampling"
    return paths


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred):
    if pred.empty or set(pred.y_true) != set(CLASSES) or not set(pred.y_pred).issubset(CLASSES):
        raise ValueError("Invalid prediction labels or class support")
    p, r, f, s = precision_recall_fscore_support(pred.y_true, pred.y_pred, labels=CLASSES, zero_division=0)
    summary = dict(accuracy=accuracy_score(pred.y_true, pred.y_pred), macro_f1=float(f.mean()),
                   weighted_f1=float(np.average(f, weights=s)), support=int(s.sum()))
    classes = pd.DataFrame(dict(candidate_class=CLASSES, precision=p, recall=r, f1=f, support=s))
    return summary, classes, confusion_matrix(pred.y_true, pred.y_pred, labels=CLASSES)


def compare(actual, saved, keys, values, name):
    a = actual.set_index(keys).sort_index()
    b = saved.set_index(keys).sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise ValueError(f"{name}: incomplete or duplicate saved metric keys")
    if not np.allclose(a[values].to_numpy(dtype=float), b[values].to_numpy(dtype=float), rtol=1e-9, atol=1e-11):
        raise ValueError(f"{name}: metrics disagree with saved predictions")


def prepare(root):
    inputs = {}
    def read(relative):
        path = root / relative
        inputs[relative] = sha(path)
        return pd.read_csv(path, keep_default_na=False, dtype={"flow_id": str})
    cohort = read("results/ml_readiness/common_cohort_manifest.csv")
    flows = read("data/processed/all_flows.csv")
    features = read("data/processed/all_features.csv")
    assigned = read("results/capture_splits/flow_partition_manifest.csv")
    assigned = assigned[assigned.design == "outer4_inner_holdout"].copy()
    if len(cohort) != 851 or cohort.flow_id.duplicated().any() or cohort.capture_id.nunique() != 35 or len(flows) != 13314 or flows.flow_id.duplicated().any():
        raise ValueError("Unexpected original cohort or flow population")
    if len(assigned) != 4*851 or assigned.duplicated(["round", "flow_id"]).any():
        raise ValueError("Invalid flow partitions")
    for fold in (1, 2, 3, 4):
        a = assigned[assigned["round"] == fold]
        if set(a.flow_id) != set(cohort.flow_id) or a.groupby("capture_id").partition.nunique().gt(1).any():
            raise ValueError("Capture leakage or incomplete fold")
        counts = a[a.partition == "train"].candidate_class.value_counts()
        if set(counts[counts > counts.median()].index) != {"MAIL", "VOIP"} or set(counts[counts < counts.median()].index) != {"CHAT", "FT"}:
            raise ValueError("Training-derived majority/minority definitions changed")
    expected = assigned[assigned.partition == "test"][["flow_id", "capture_id", "candidate_class", "round", "verification_status", "label_source"]].rename(columns={"candidate_class": "y_true", "round": "fold"})
    pooled, classes, folds, foldclasses, confusions, inventory = [], [], [], [], [], []
    for (model, method), folder in sources().items():
        # This explicit allowlist never reads historical bootstrap-style ROS.
        if "/oversampling_exploratory" in folder:
            raise ValueError("Historical defective ROS cannot be a paper source")
        pred = read(folder + "/predictions.csv")
        if "model" in pred:
            pred = pred[pred.model == model].copy()
        if len(pred) != 3404 or pred.duplicated(["flow_id", "n_packets"]).any() or set(pred.n_packets) != set(BUDGETS):
            raise ValueError(f"{model}/{method}: invalid prediction coverage")
        pp, cc, ff, fc, mm = [], [], [], [], []
        for n in BUDGETS:
            sub = pred[pred.n_packets == n]
            identity = list(expected.columns)
            if len(sub) != 851 or not sub[identity].sort_values("flow_id").reset_index(drop=True).equals(expected.sort_values("flow_id").reset_index(drop=True)):
                raise ValueError("Original test identities, labels, captures or folds changed")
            met, perclass, matrix = metrics(sub)
            pp.append(dict(n_packets=n, **met))
            cc.append(perclass.assign(n_packets=n))
            for f in (1, 2, 3, 4):
                metf, clf, _ = metrics(sub[sub.fold == f])
                ff.append(dict(n_packets=n, fold=f, **metf))
                fc.append(clf.assign(n_packets=n, fold=f))
            for i, true in enumerate(CLASSES):
                for j, predicted in enumerate(CLASSES):
                    mm.append(dict(n_packets=n, true_class=true, predicted_class=predicted,
                                   count=int(matrix[i, j]), row_pct=100*matrix[i, j]/matrix[i].sum()))
        pp, cc, ff, fc, mm = pd.DataFrame(pp), pd.concat(cc), pd.DataFrame(ff), pd.concat(fc), pd.DataFrame(mm)
        legacy_weighted = folder == "results/ml/weighted_exploratory"
        for filename, actual, keys, values in [
            ("weighted_metrics.csv" if legacy_weighted else "pooled_metrics.csv", pp, ["n_packets"], ["accuracy", "macro_f1", "weighted_f1", "support"]),
            ("weighted_per_class.csv" if legacy_weighted else "pooled_class_metrics.csv", cc, ["n_packets", "candidate_class"], ["precision", "recall", "f1", "support"]),
        ]:
            saved = read(folder + "/" + filename)
            if "model" in saved:
                saved = saved[saved.model == model]
            compare(actual, saved, keys, values, filename)
        if not legacy_weighted:
            compare(ff, read(folder + "/fold_metrics.csv"), ["n_packets", "fold"], ["accuracy", "macro_f1", "weighted_f1", "support"], "fold metrics")
            compare(fc, read(folder + "/fold_class_metrics.csv"), ["n_packets", "fold", "candidate_class"], ["precision", "recall", "f1", "support"], "fold class metrics")
            saved = read(folder + "/pooled_confusion.csv").rename(columns={"actual_class": "true_class"})
            compare(mm, saved, ["n_packets", "true_class", "predicted_class"], ["count"], "confusion counts")
        pooled.append(pp.assign(model=model, method=method))
        classes.append(cc.assign(model=model, method=method))
        folds.append(ff.assign(model=model, method=method))
        foldclasses.append(fc.assign(model=model, method=method))
        confusions.append(mm.assign(model=model, method=method))
        inventory.append(dict(model=model, method=method, directory=folder, predictions=len(pred), flows_per_n=851, folds=4))
    pooled, classes, folds = pd.concat(pooled, ignore_index=True), pd.concat(classes, ignore_index=True), pd.concat(folds, ignore_index=True)
    role = classes.copy()
    role["role"] = role.candidate_class.map({"CHAT": "Minority", "FT": "Minority", "MAIL": "Majority", "VOIP": "Majority"})
    roles = role.groupby(["model", "method", "n_packets", "role"], sort=False).agg(mean_pooled_class_f1=("f1", "mean"), support=("support", "sum")).reset_index()
    fold_summary = folds.groupby(["model", "method", "n_packets"], sort=False).macro_f1.agg(fold_mean="mean", fold_sd="std", fold_min="min", fold_max="max").reset_index()
    delay = read("results/supervised_delay/supervised_delay_overall.csv")
    selected = features[features.flow_id.isin(cohort.flow_id)].copy()
    if len(selected) != 3404 or selected.duplicated(["flow_id", "n_packets"]).any() or not selected.groupby("flow_id").n_packets.nunique().eq(4).all():
        raise ValueError("Delay cohort representations incomplete")
    selected["delay_ms"] = selected.time_to_n*1000
    if not np.isfinite(selected.delay_ms).all() or selected.delay_ms.lt(0).any():
        raise ValueError("Invalid acquisition delays")
    calculated = selected.groupby("n_packets").delay_ms.agg(flows="size", mean_ms="mean", median_ms="median", p90_ms=lambda s: s.quantile(.9), p95_ms=lambda s: s.quantile(.95)).reset_index()
    compare(calculated, delay, ["n_packets"], ["flows", "mean_ms", "median_ms", "p90_ms", "p95_ms"], "supervised acquisition delay")
    coverage = pd.DataFrame([dict(n_packets=n, eligible_flows=int(flows.packet_count.ge(n).sum()), total_extracted_flows=len(flows), coverage_pct=100*flows.packet_count.ge(n).mean()) for n in BUDGETS])
    return dict(pooled_metrics=pooled, pooled_class_metrics=classes, majority_minority=roles,
                fold_metrics=folds, fold_class_metrics=pd.concat(foldclasses, ignore_index=True),
                fold_summary=fold_summary, pooled_confusions=pd.concat(confusions, ignore_index=True),
                supervised_acquisition_delay=calculated, flow_coverage=coverage,
                source_inventory=pd.DataFrame(inventory)), inputs


def style(ax, ylabel="Pooled outer-test macro-F1"):
    ax.set_xticks(BUDGETS)
    ax.set_xlim(2, 21)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Observed packets (N)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)


def lines(ax, frame, value):
    for method in METHODS:
        sub = frame[frame.method == method].sort_values("n_packets")
        ax.plot(sub.n_packets, sub[value], color=COLORS[method], marker=MARKERS[method], linestyle=STYLES[method], label=method)


def legend(fig, title=None):
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle=STYLES[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, title=title)


def render(tables, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 9, "lines.linewidth": 1.8,
                         "lines.markersize": 5, "pdf.fonttype": 42, "ps.fonttype": 42})
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
        plt.close(fig)
    pooled, classes = tables["pooled_metrics"], tables["pooled_class_metrics"]
    fig, ax = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
    for model, color, marker in zip(MODELS, COLORS.values(), MARKERS.values()):
        sub = pooled[(pooled.model == model) & (pooled.method == "Baseline")].sort_values("n_packets")
        ax.plot(sub.n_packets, sub.macro_f1, color=color, marker=marker, label=NAMES[model])
    style(ax)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig01_baseline_macro_f1")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True, layout="constrained")
    for ax, model in zip(axes, MODELS):
        lines(ax, pooled[pooled.model == model], "macro_f1")
        style(ax)
        ax.set_title(NAMES[model])
    legend(fig)
    save(fig, "fig02_methods_macro_f1")
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), sharex=True, sharey=True, layout="constrained")
    for row, model in enumerate(MODELS):
        for col, role in enumerate(("Majority", "Minority")):
            ax = axes[row, col]
            sub = tables["majority_minority"]
            lines(ax, sub[(sub.model == model) & (sub.role == role)], "mean_pooled_class_f1")
            style(ax, "Mean of two pooled class F1 scores")
            ax.set_title(f"{NAMES[model]}: {role}\n" + ("MAIL and VOIP" if role == "Majority" else "CHAT and FT"))
    legend(fig)
    save(fig, "fig03_majority_minority_f1")
    for model in MODELS:
        fig, axes = plt.subplots(2, 2, figsize=(8.7, 6.3), sharex=True, sharey=True, layout="constrained")
        fig.suptitle(NAMES[model])
        for ax, cls in zip(axes.flat, CLASSES):
            lines(ax, classes[(classes.model == model) & (classes.candidate_class == cls)], "f1")
            style(ax, "Pooled outer-test class F1")
            ax.set_title(cls)
        legend(fig)
        save(fig, f"fig04_{model}_per_class_f1")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8), layout="constrained")
    for row, model in enumerate(("random_forest", "xgboost")):
        for col, n in enumerate((5, 20)):
            ax = axes[row, col]
            sub = tables["pooled_confusions"]
            sub = sub[(sub.model == model) & (sub.method == "Baseline") & (sub.n_packets == n)]
            counts = sub.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=CLASSES, columns=CLASSES).to_numpy()
            pct = 100*counts/counts.sum(axis=1, keepdims=True)
            im = ax.pcolormesh(np.arange(5)-.5, np.arange(5)-.5, pct, cmap="Blues", vmin=0, vmax=100)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{counts[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center", color="white" if pct[i,j] > 55 else "black", fontsize=10)
            ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
                   xlabel="Predicted class", ylabel="True class", title=f"{NAMES[model]}, N={n}")
    colorbar = fig.colorbar(im, ax=axes, shrink=.8, label="Percentage within true class (%)")
    colorbar.solids.set_rasterized(False)
    save(fig, "fig05_baseline_confusions")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True, sharey=True, layout="constrained")
    fold_styles = ("-", "--", "-.", ":")
    fold_markers = ("o", "s", "^", "D")
    for row, model in enumerate(MODELS):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            sub = tables["fold_metrics"]
            for fold, ls, marker in zip((1, 2, 3, 4), fold_styles, fold_markers):
                f = sub[(sub.model == model) & (sub.method == method) & (sub.fold == fold)].sort_values("n_packets")
                ax.plot(f.n_packets, f.macro_f1, color=COLORS[method], linestyle=ls, marker=marker, alpha=.85, linewidth=1.4)
            style(ax, "Outer-fold macro-F1")
            ax.set_title(f"{NAMES[model]}\n{method}")
    handles = [Line2D([], [], color="#444444", linestyle=ls, marker=marker, label=f"Fold {f}") for f, ls, marker in zip((1,2,3,4), fold_styles, fold_markers)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, title="Descriptive fold variation; not confidence intervals")
    save(fig, "fig06_fold_macro_f1")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    delay, coverage = tables["supervised_acquisition_delay"], tables["flow_coverage"]
    ax = axes[0]
    for col, label, color, marker in (("median_ms", "Median", "#0072B2", "o"), ("p95_ms", "95th percentile", "#D55E00", "s")):
        ax.plot(delay.n_packets, delay[col], label=label, color=color, marker=marker)
    ax.set_yscale("symlog", linthresh=1)
    ax.set(xlabel="Observed packets (N)", ylabel="Packet acquisition time (ms; symlog scale)", title="Fixed supervised cohort: 851 flows", xticks=BUDGETS)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.3)
    ax = axes[1]
    ax.plot(coverage.n_packets, coverage.coverage_pct, marker="o", color="#0072B2")
    for r in coverage.itertuples():
        align = "left" if r.n_packets == 3 else "right" if r.n_packets == 20 else "center"
        ax.annotate(f"{r.eligible_flows:,} ({r.coverage_pct:.2f}%)", (r.n_packets, r.coverage_pct), xytext=(0,-18 if r.n_packets == 20 else 8), textcoords="offset points", ha=align, fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1))
    ax.set(xlabel="Required packets (N)", ylabel="Eligible flows / 13,314 extracted flows (%)", title="Full extracted population: 13,314 flows", xticks=BUDGETS, ylim=(0,32), xlim=(1,23))
    ax.grid(axis="y", alpha=.3)
    save(fig, "fig07_acquisition_delay_coverage")


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing work: {output}. Use --output with a new directory.")
    if not output.is_relative_to(root / "results/ml/paper_figures"):
        raise ValueError("Output must be results/ml/paper_figures or a new directory beneath it")
    protected = {p: sha(p) for base in (root / "results", root / "data/processed") for p in base.rglob("*") if p.is_file()}
    tables, inputs = prepare(root)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    render(tables, output)
    for path, digest in protected.items():
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"Protected source changed: {path}")
    script = Path(__file__).resolve()
    validation = dict(status="validated", analysis_only=True, models=3, methods=3, packet_budgets=BUDGETS,
                      combinations=9, pooled_points=36, pooled_class_points=144, fold_points=144,
                      evaluated_flows=851, captures=35, extracted_flow_denominator=13314,
                      protected_files_unchanged=len(protected), input_sha256=inputs,
                      script_sha256=sha(script), versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__),
                      warning="Weak capture labels; unknown recording independence; exploratory comparisons; no confidence intervals")
    text = ["# Final paper figures", "", "Analysis only: no model fitting, tuning, resplitting or dataset modification.", "",
            "## Figure inventory", ""]
    text += [f"- `{name}.png` and `{name}.pdf`: {purpose}" for name, purpose in FIGURES.items()]
    text += ["", "PNG exports are 300 DPI; PDF exports retain vector charts and embedded TrueType fonts.",
             "Lines connect observed budgets only; no smoothing or unsupported budget estimates are generated.", "", "## Exact experiment inputs", ""]
    text += [f"- {NAMES[m]} / {method}: `{folder}/predictions.csv`" for (m, method), folder in sources().items()]
    text += ["", "Saved pooled/class/fold metrics and baseline/new confusion tables are cross-checked; `validation.json` lists every read input and its hash.",
             "Historical `results/ml/oversampling_exploratory/` is never read. Historical LR/RF weighting has weaker execution provenance than the new runs.",
             "", "## Aggregation and supporting tables", "",
             "All model comparisons use the same 851 flows at every N and original four-fold test assignments. Class order is CHAT, MAIL, VOIP, FT; supports are 92, 247, 389, 123.",
             "`pooled_metrics.csv` contains 36 pooled outer-test summaries. `pooled_class_metrics.csv` contains 144 precision/recall/F1/support rows.",
             "`majority_minority.csv` averages the two pooled class F1 scores: MAIL/VOIP majority and CHAT/FT minority. Roles are verified against original training counts. These are not fold-averaged role scores.",
             "`fold_metrics.csv`, `fold_class_metrics.csv` and `fold_summary.csv` retain individual folds and descriptive mean/sample SD/min/max. Fold SD/range is not a confidence interval; fold training pools overlap.",
             "`pooled_confusions.csv` contains counts and percentages normalized within each true class.",
             "`supervised_acquisition_delay.csv` validates `results/supervised_delay/supervised_delay_overall.csv` against `data/processed/all_features.csv` and the 851-flow cohort. Seconds are converted to milliseconds; quantiles use linear interpolation. Acquisition time excludes inference/processing latency.",
             "`flow_coverage.csv` uses packet counts in `data/processed/all_flows.csv`; denominator is all 13,314 flows, including unresolved labels and short flows. Evaluation-cohort support remains 851 at every N.",
             "`source_inventory.csv` records the nine approved source mappings. Identity/provenance checks use `results/ml_readiness/common_cohort_manifest.csv` and `results/capture_splits/flow_partition_manifest.csv`.",
             "", "## Scientific limits", "",
             "Capture-derived labels are unverified; captures are not proven independent sessions. MAIL/FT have four captures each and archive/class confounding. The cohort selects flows reaching 20 packets. Later balancing comparisons reuse inspected tests and frozen baseline parameters; observed differences are descriptive, not significance claims. Historical preprocessing provenance remains uncertain.",
             "", "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe -B -m src.analysis.paper_figures", "```", "",
             "The command refuses an existing output directory. To regenerate while preserving all prior figures:", "", "```powershell",
             ".\\.venv\\Scripts\\python.exe -B -m src.analysis.paper_figures --output results/ml/paper_figures/regenerated-01", "```", "",
             "Choose another new subdirectory for subsequent regenerations. No experimental outputs are overwritten."]
    (output / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    validation["output_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), figures=len(FIGURES), png=len(FIGURES), pdf=len(FIGURES), tables=len(tables), protected_files_unchanged=len(protected), status="validated"), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/ml/paper_figures")
    args = parser.parse_args()
    run(ROOT, args.output)


if __name__ == "__main__":
    main()
