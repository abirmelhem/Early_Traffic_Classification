"""Compare completed, compatible result files only. Never import model backends."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .data_loader import ROOT
from .features import CLASSES
from .models import MODELS


def load_results(root, models=tuple(MODELS)):
    available, missing = {}, []
    for model in models:
        folder = Path(root) / "results/ml" / model
        if not folder.exists():
            missing.append(model)
            continue
        marker = folder / "completed.json"
        if not marker.is_file():
            raise ValueError(f"Incomplete experiment directory: {folder}")
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or completed.get("model") != model:
            raise ValueError("Invalid experiment completion marker")
        for name, digest in completed["output_sha256"].items():
            if Path(name).name != name or hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Result checksum mismatch")
        config = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(folder / "pooled_metrics.csv")
        per_class = pd.read_csv(folder / "pooled_class_metrics.csv")
        predictions = pd.read_csv(folder / "predictions.csv", dtype={"flow_id": str})
        if predictions.duplicated(["flow_id", "n_packets"]).any():
            raise ValueError("Duplicate pooled prediction identity")
        for col in ["accuracy", "macro_f1", "weighted_f1"]:
            if not np.isfinite(metrics[col]).all() or not metrics[col].between(0, 1).all():
                raise ValueError("Invalid metric range")
        available[model] = dict(config=config, metrics=metrics, per_class=per_class, predictions=predictions)
    identity = None
    reference = None
    for result in available.values():
        if identity is None:
            identity = result["config"]["comparison_identity"]
        if result["config"]["comparison_identity"] != identity:
            raise ValueError("Incompatible inputs, protocol, seed or packet budgets across models")
        keys = result["predictions"][["n_packets", "fold", "flow_id", "capture_id", "y_true"]].sort_values(["n_packets", "flow_id"]).reset_index(drop=True)
        if reference is None:
            reference = keys
        elif not reference.equals(keys):
            raise ValueError("Different held-out identities or labels across model results")
    return available, missing


def compare(root=ROOT):
    results, missing = load_results(root)
    if missing:
        warnings.warn("Missing models (not retrained): " + ", ".join(missing), UserWarning)
    if not results:
        print("No completed model results. No comparison files or plots generated.")
        return None
    output = Path(root) / "results/ml/comparisons"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparisons: {output}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = pd.concat([r["metrics"].assign(model=name) for name, r in results.items()], ignore_index=True)
    per_class = pd.concat([r["per_class"].assign(model=name) for name, r in results.items()], ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "pooled_model_comparison.csv", index=False)
    per_class.to_csv(output / "per_class_model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in summary.groupby("model"):
        group = group.sort_values("n_packets")
        ax.plot(group.n_packets, group.macro_f1, marker="o", label=name)
    ax.set(xlabel="Observed packets N", ylabel="Pooled outer-test macro-F1", ylim=(0, 1), xticks=sorted(summary.n_packets.unique()),
           title="Capture-disjoint evaluation; unverified candidate labels")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "macro_f1_by_n.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for cls, ax in zip(CLASSES, axes.flat):
        for name, group in per_class[per_class.candidate_class.eq(cls)].groupby("model"):
            group = group.sort_values("n_packets")
            ax.plot(group.n_packets, group.f1, marker="o", label=name)
        ax.set(title=cls, xlabel="N", ylabel="Pooled per-class F1", ylim=(0, 1), xticks=sorted(summary.n_packets.unique()))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "per_class_f1_by_n.png", dpi=300); plt.close(fig)
    (output / "README.md").write_text(
        "# Comparison of existing completed results\n\n"
        "Capture-derived unverified targets; recording independence is unverified. "
        "Pooled F1 is not mean fold F1. No uncertainty intervals are implied. "
        "Do not select a model or N from outer test scores.\n\n"
        f"Available: {', '.join(results)}. Missing: {', '.join(missing) or 'none'}.\n", encoding="utf-8")
    print(f"Compared {len(results)} existing model result sets: {output}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    compare(args.root)


if __name__ == "__main__":
    main()
