"""Post-hoc saved-result analysis. No estimator construction, fitting or resplitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.compare_models import load_results
from src.ml.data_loader import ROOT
from src.ml.features import CLASSES, BUDGETS, DESIGN
from src.ml.metrics import calculate_metrics
from src.analysis.dataset_eda import markdown


def verify_metrics(predictions, pooled, folds):
    for n in BUDGETS:
        sub = predictions[predictions.n_packets == n]
        _, actual, _ = calculate_metrics(sub.y_true, sub.y_pred)
        saved = pooled[pooled.n_packets == n].set_index("candidate_class").reindex(CLASSES)
        for col in ["precision", "recall", "f1", "support"]:
            if not np.allclose(actual[col], saved[col], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Saved pooled {col} disagrees with predictions")
        for fold in (1, 2, 3, 4):
            test = sub[sub.fold == fold]
            _, actual, _ = calculate_metrics(test.y_true, test.y_pred)
            saved = folds[(folds.n_packets == n) & (folds.fold == fold)].set_index("candidate_class").reindex(CLASSES)
            for col in ["precision", "recall", "f1", "support"]:
                if not np.allclose(actual[col], saved[col], rtol=1e-8, atol=1e-10):
                    raise ValueError(f"Saved fold {col} disagrees with predictions")


def class_changes(pooled):
    rows = []
    for (model, cls), sub in pooled.groupby(["model", "candidate_class"]):
        values = sub.set_index("n_packets").f1
        maximum = values.max()
        rows.append(dict(model=model, candidate_class=cls, f1_n3=values[3], f1_n5=values[5],
                         f1_n10=values[10], f1_n20=values[20], delta_3_to_20=values[20]-values[3],
                         delta_5_to_10=values[10]-values[5], delta_5_to_20=values[20]-values[5],
                         best_observed_n=";".join(str(int(n)) for n, v in values.items() if np.isclose(v, maximum, rtol=0, atol=1e-12)),
                         best_observed_f1=maximum))
    return pd.DataFrame(rows)


def role_summary(folds, roles):
    joined = folds.merge(roles, on=["model", "n_packets", "fold", "candidate_class"], validate="one_to_one")
    if joined.role.isna().any(): raise ValueError("Missing training-frequency role")
    per_fold = joined.groupby(["model", "n_packets", "fold", "role"]).agg(
        mean_class_f1=("f1", "mean"), evaluated_flows=("support", "sum"), classes=("candidate_class", "size")).reset_index()
    summary = per_fold.groupby(["model", "n_packets", "role"]).mean_class_f1.agg(
        fold_mean="mean", fold_std="std", fold_min="min", fold_max="max").reset_index()
    return joined, per_fold, summary


def prepare(root):
    results, missing = load_results(root)
    if missing: raise ValueError(f"Expected three completed models; missing: {missing}")
    pooled, folds, roles, confusion, transitions, capture_errors, selected = [], [], [], [], [], [], []
    assignments = pd.read_csv(root / "results/capture_splits/flow_partition_manifest.csv", keep_default_na=False)
    assignments = assignments[assignments.design == DESIGN]
    for model, result in results.items():
        folder = root / "results/ml" / model
        for relative, digest in result["config"]["comparison_identity"]["inputs"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Input changed since model run: {relative}")
        p = result["per_class"].assign(model=model)
        f = pd.read_csv(folder / "fold_class_metrics.csv").assign(model=model)
        r = pd.read_csv(folder / "training_class_roles.csv").assign(model=model)
        verify_metrics(result["predictions"], p, f)
        for (n, fold), group in r.groupby(["n_packets", "fold"]):
            train = assignments[(assignments["round"] == fold) & (assignments.partition == "train")]
            counts = train.candidate_class.value_counts().reindex(CLASSES)
            median = counts.median()
            for row in group.itertuples():
                expected = "majority" if counts[row.candidate_class] > median else "minority" if counts[row.candidate_class] < median else "middle"
                if row.role != expected or row.training_count != counts[row.candidate_class]:
                    raise ValueError("Stored training-frequency role is inconsistent")
        pooled.append(p); folds.append(f); roles.append(r)
        pred = result["predictions"]
        for n, group in pred.groupby("n_packets"):
            _, _, matrix = calculate_metrics(group.y_true, group.y_pred)
            saved = pd.read_csv(folder / "pooled_confusion.csv")
            saved = saved[saved.n_packets == n].pivot(index="actual_class", columns="predicted_class", values="count").reindex(index=CLASSES, columns=CLASSES)
            if not np.array_equal(matrix, saved.to_numpy()): raise ValueError("Confusion counts disagree")
            for i, actual in enumerate(CLASSES):
                for j, predicted in enumerate(CLASSES):
                    if i != j:
                        confusion.append(dict(model=model, n_packets=n, actual_class=actual, predicted_class=predicted,
                                              errors=int(matrix[i, j]), true_class_support=int(matrix[i].sum()),
                                              true_class_error_pct=100*matrix[i, j]/matrix[i].sum()))
        a = pred[pred.n_packets == 5].set_index("flow_id")
        b = pred[pred.n_packets == 20].set_index("flow_id").reindex(a.index)
        for cls in CLASSES:
            mask = a.y_true.eq(cls)
            correct5 = a.loc[mask, "y_pred"].eq(cls)
            correct20 = b.loc[mask, "y_pred"].eq(cls)
            transitions.append(dict(model=model, candidate_class=cls, support=int(mask.sum()),
                                    correct_at_both=int((correct5 & correct20).sum()),
                                    wrong_at_both=int((~correct5 & ~correct20).sum()),
                                    correct5_wrong20=int((correct5 & ~correct20).sum()),
                                    wrong5_correct20=int((~correct5 & correct20).sum())))
        for (n, fold, capture, cls), group in pred.groupby(["n_packets", "fold", "capture_id", "y_true"]):
            capture_errors.append(dict(model=model, n_packets=n, fold=fold, capture_id=capture,
                                       candidate_class=cls, support=len(group), errors=int(group.y_pred.ne(cls).sum()),
                                       recall=float(group.y_pred.eq(cls).mean())))
        selected.append(pd.read_csv(folder / "selected_parameters.csv").assign(model=model))
    pooled = pd.concat(pooled, ignore_index=True)
    folds = pd.concat(folds, ignore_index=True)
    roles = pd.concat(roles, ignore_index=True)
    joined, role_folds, role_aggregate = role_summary(folds, roles)
    variability = folds.groupby(["model", "candidate_class", "n_packets"]).agg(
        fold_mean_f1=("f1", "mean"), fold_std_f1=("f1", "std"), fold_min_f1=("f1", "min"),
        fold_max_f1=("f1", "max"), test_support_min=("support", "min"), test_support_max=("support", "max")).reset_index()
    changes = class_changes(pooled)
    macro = pd.concat([r["metrics"].assign(model=m) for m,r in results.items()], ignore_index=True)
    best_model = macro.loc[macro.groupby("model").macro_f1.idxmax(), ["model", "n_packets", "macro_f1"]].rename(columns={"n_packets":"best_observed_n"})
    errors = pd.DataFrame(confusion)
    top_errors = errors.sort_values(["model", "n_packets", "errors", "true_class_error_pct"], ascending=[True,True,False,False]).groupby(["model", "n_packets"]).head(3)
    delay = pd.read_csv(root / "results/eda/tables/delay_common_n20_cohort.csv")
    specific_delay = pd.read_csv(root / "results/eda/tables/delay_by_n.csv")
    coverage = pd.read_csv(root / "results/eda/tables/class_by_n.csv")
    return dict(pooled_per_class=pooled, class_f1_changes=changes, fold_class_metrics=folds,
                fold_variability=variability, training_roles=roles, role_fold_metrics=role_folds,
                role_summary=role_aggregate, model_best_budget=best_model,
                directional_confusions=errors, top_confusions=top_errors,
                paired_n5_n20_correctness=pd.DataFrame(transitions), capture_errors=pd.DataFrame(capture_errors),
                selected_parameters=pd.concat(selected,ignore_index=True), macro_metrics=macro,
                delay_common_1107=delay, delay_n_specific=specific_delay, coverage=coverage)


def plots(t, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(CLASSES, ["#228833", "#AA3377", "#EE7733", "#332288"]))
    models = list(t["pooled_per_class"].model.unique())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    def save(fig, name):
        fig.tight_layout();fig.savefig(folder / f"{name}.png", dpi=300, facecolor="white");plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for model,ax in zip(models,axes):
        for cls in CLASSES:
            s=t["pooled_per_class"].query("model == @model and candidate_class == @cls").sort_values("n_packets")
            ax.plot(s.n_packets,s.f1,marker="o",label=cls,color=colors[cls])
        ax.set(title=model,xlabel="Packets N",ylabel="Pooled outer-test class F1",xticks=BUDGETS,ylim=(0,1));ax.legend(fontsize=8)
    save(fig,"pooled_class_f1")
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharey=True)
    for i,model in enumerate(models):
        for j,cls in enumerate(CLASSES):
            ax=axes[i,j]
            s=t["fold_class_metrics"].query("model == @model and candidate_class == @cls")
            for fold,g in s.groupby("fold"):
                g=g.sort_values("n_packets");ax.plot(g.n_packets,g.f1,marker=".",alpha=.6,label=f"Fold {fold}")
            ax.set(title=f"{model}: {cls}",xlabel="N",ylabel="Fold class F1",xticks=BUDGETS,ylim=(0,1))
            if i==0 and j==0:ax.legend(fontsize=7)
    save(fig,"fold_class_variability")
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for model,ax in zip(models,axes):
        for role,g in t["role_summary"].query("model == @model").groupby("role"):
            g=g.sort_values("n_packets");ax.plot(g.n_packets,g.fold_mean,marker="o",label=role)
            ax.fill_between(g.n_packets,g.fold_min,g.fold_max,alpha=.12)
        ax.set(title=model,xlabel="N",ylabel="Mean fold class-F1 within role",xticks=BUDGETS,ylim=(0,1));ax.legend()
    fig.suptitle("Training-derived roles; shading = fold range, not confidence interval")
    save(fig,"training_role_f1")
    fig,axes=plt.subplots(3,2,figsize=(10,12))
    for i,model in enumerate(models):
        for j,n in enumerate([5,20]):
            s=t["pooled_per_class"].query("model == @model and n_packets == @n").set_index("candidate_class")
            err=t["directional_confusions"].query("model == @model and n_packets == @n")
            counts=np.zeros((4,4),dtype=int)
            for a,cls in enumerate(CLASSES):
                counts[a,a]=round(s.loc[cls,"recall"]*s.loc[cls,"support"])
                for b,other in enumerate(CLASSES):
                    if a!=b:counts[a,b]=int(err.query("actual_class == @cls and predicted_class == @other").errors.iloc[0])
            pct=counts/counts.sum(axis=1,keepdims=True)*100
            ax=axes[i,j];ax.imshow(pct,vmin=0,vmax=100,cmap="Blues")
            for a in range(4):
                for b in range(4):ax.text(b,a,f"{counts[a,b]}\n{pct[a,b]:.1f}%",ha="center",va="center",fontsize=8,color="white" if pct[a,b]>55 else "black")
            ax.set(title=f"{model}, N={n}",xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,xlabel="Predicted",ylabel="Actual")
    save(fig,"confusions_n5_n20")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    d=t["delay_common_1107"]
    axes[0].plot(d.n_packets,d.median_ms,marker="o",label="Median")
    axes[0].plot(d.n_packets,d.p95_ms,marker="o",label="95th percentile")
    axes[0].set(yscale="log",title="Fixed EDA cohort: 1,107 flows (includes unresolved)",xlabel="N",ylabel="Observation delay, ms (log)",xticks=BUDGETS);axes[0].legend()
    for cls,g in t["coverage"].groupby("candidate_class"):
        if g.total_flows.max()>0:axes[1].plot(g.n_packets,g.coverage_pct,marker="o",label=cls)
    axes[1].set(title="Within-candidate eligibility, full dataset",xlabel="N",ylabel="Coverage (%)",xticks=BUDGETS,ylim=(0,100));axes[1].legend(fontsize=7)
    save(fig,"delay_and_coverage")


def report(t):
    best=t["model_best_budget"]
    var=t["fold_variability"].query("n_packets == 20")
    text="""# Completed experiment analysis

## Scope and estimands

Three completed models, four packet budgets, four capture-disjoint outer folds.
Every model/N pools one held-out prediction for each of the same 851 flows: CHAT 92,
MAIL 247, VOIP 389, FT 123. Inputs and saved result checksums were validated; class
metrics and confusion counts were recomputed from predictions without fitting.
Targets are unverified capture-derived candidates. No model, dataset, split, or
existing result is modified by this analysis.

Pooled metrics concatenate all outer-test predictions before computing precision,
recall and F1. Fold means average four separately calculated metrics without weighting
by fold size. They are different estimands: F1 is nonlinear and test supports differ.
Fold SD/range are descriptive variability, not standard errors or confidence intervals;
folds share training data. No statistical significance or optimality is claimed.

## Overall pooled results and observed budgets

"""+markdown(t["macro_metrics"][["model","n_packets","accuracy","macro_f1","weighted_f1","support"]])+"\n\n"+markdown(best)
    text+="""

These are retrospective maxima on saved test results, not validation-based choices
or authorization to tune using test outcomes. The primary comparison remains all N.

## Per-class precision, recall, F1 and support — pooled outer tests

"""+markdown(t["pooled_per_class"][["model","n_packets","candidate_class","precision","recall","f1","support"]].sort_values(["model","n_packets","candidate_class"]))
    text+="\n\n## Class-specific changes and best observed N\n\n"+markdown(t["class_f1_changes"])
    text+="""

Logistic Regression's FT, MAIL and VOIP improve from N=3 to N=20; CHAT remains weak
and is nonmonotonic, with its maximum at N=5. Random Forest peaks at N=5 for all
four classes and declines thereafter. XGBoost peaks at N=5 for CHAT/MAIL/FT, while
VOIP's maximum is N=10. Thus increased observation does not uniformly improve these
aggregate-feature models. This is not evidence that minority classes universally
need more packets: FT improves strongly for Logistic Regression but deteriorates
after N=5 for both tree ensembles.

## Fold variability and capture concentration

N=20 snapshot (full N-by-class table in tables/fold_variability.csv):

"""+markdown(var)
    text+="""

CHAT has only 22–24 test flows per fold. MAIL and FT each have one test capture per
fold: MAIL support ranges 32–94 and FT 23–44. Their fold metrics therefore confound
capture variation with model performance. For example, Random Forest MAIL N=20
pooled F1 is about .860 but its unweighted fold mean is about .759; large, easier
captures contribute more to the pooled result. FT N=20 ranges about .238–.936 for
Random Forest and .204–.957 for XGBoost. These ranges undermine a precise general
claim based only on pooled F1. Inspect tables/capture_errors.csv for the actual
held-out captures driving errors. Capture-level recall there is not per-capture F1:
false positives from other target captures are not available in that subgroup.

![Fold class metrics](plots/fold_class_variability.png)

## Majority/minority comparison — derived from training only

For each fold, above/below the median of its four original training class counts
defines majority/minority (ties middle). Stored roles were checked against the saved
training assignments. MAIL and VOIP are majority in all four rounds; CHAT and FT
are minority. Roles remain fixed across N and are not redefined using test frequencies
or the subsequent train+validation refit. Thus MAIL is a majority candidate in this
cohort even though it has only four captures; flow frequency and recording diversity
are different concepts.

The following are equal-class F1 means within each role, then unweighted means over
folds. They are not a pooled binary majority/minority classification score.

"""+markdown(t["role_summary"])
    text+="""

![Training-role performance](plots/training_role_f1.png)

## Directional confusions

Largest three off-diagonal counts per model/N follow. Percentages use true-class
support, not all mistakes; high counts can reflect larger class support. Full
directional tables and row-normalized heatmaps are supplied.

At N=20 the largest ensemble confusions are FT→VOIP and MAIL→VOIP. Random Forest
FT→VOIP grows from 11/123 (8.9%) at N=5 to 44/123 (35.8%) at N=20; MAIL→VOIP
grows from 5/247 to 37/247. XGBoost FT→VOIP grows from 16/123 to 45/123 and
MAIL→VOIP from 5/247 to 36/247. These are observed directional errors, not causal
explanations. VOIP recall actually rises slightly between N=5 and N=20 for both
ensembles, while its F1 declines because precision deteriorates as other classes
are increasingly predicted as VOIP. Thus a lower class F1 need not mean lower recall.

"""+markdown(t["top_confusions"])
    text+="""

![N=5 versus N=20 confusion](plots/confusions_n5_n20.png)

## Why do tree ensembles decline after N=5?

**Evidence:** the same test identities and class supports occur at every N, so this
primary decline is not caused by fewer test flows or changing class composition.
All Random Forest classes decline from N=5 to N=20, with FT the largest F1 loss.
XGBoost FT likewise has the largest loss; VOIP improves slightly at N=10 before
declining. Paired prediction transitions below count actual changes on the same
flows, avoiding an interpretation based solely on pooled scores.

"""+markdown(t["paired_n5_n20_correctness"])
    text+="""

**Hypotheses, not established causes:** later packets alter size/direction/IAT
aggregates and may dilute distinctive early handshake patterns; seven aggregate
features lose packet order, so additional packets are not guaranteed to add useful
separability. Capture-dependent timing, packet framing and application behavior may
generalize differently as prefixes lengthen. Limited capture diversity and noisy
capture-level targets could amplify those effects. Validation uses few captures,
so selected hyperparameters may not generalize to particular held-out recordings.
Selected settings are copied to tables/selected_parameters.csv for inspection;
different selected settings alone do not prove overfitting. The saved data cannot
establish causality, raw-flow label correctness, or recording independence. No new
training, feature analysis on raw packets, or post-hoc tuning was performed.

## Delay and coverage

Fixed EDA common cohort, **1,107** flows including unresolved traffic (not the 851-flow
supervised population):

"""+markdown(t["delay_common_1107"])
    text+="""

Median delay rises from 27.705 ms at N=3 to 462.936 ms at N=20; the long upper tail
is visible in the p95. This is observation delay, not compute/inference latency.
Existing N-specific delay summaries use changing populations and are copied separately
to tables/delay_n_specific.csv. Do not attach the 1,107-flow delay quantiles to the
851-flow supervised cohort as though they were measured on the same population.

Full extracted-population coverage is 24.027%, 16.592%, 13.302%, 8.315% for
N=3,5,10,20 respectively (13,314-flow denominator). Candidate-specific coverage is
copied to tables/coverage.csv, retaining UNRESOLVED. Primary ML support stays fixed
at 851 for every N. Coverage loss describes operational reach, not accuracy loss.

![Delay and eligibility](plots/delay_and_coverage.png)

## Limitations and conclusions

The observed tree-ensemble peak at N=5 suggests a promising reliability/delay tradeoff
on this fixed long-flow cohort, but does not establish a universally best budget.
Logistic Regression improves markedly at longer prefixes, demonstrating model-dependent
effects. Minority candidates remain harder on average, but class identity, application,
protocol, capture setting and frequency are confounded. There is no controlled causal
experiment isolating class imbalance, and four classes are not independent replicates
of majority/minority status.

Only 35 captures contribute to supervision, recording relationships are unverified,
and weak labels may include background traffic. The >=20 selection excludes most
flows, biases protocol composition and limits generalization to short traffic.
Archive-02 raw verification and extraction provenance remain outstanding. No uncertainty
intervals, independent-session guarantees, or statistical significance are invented.
Report all N and per-fold/class outcomes; do not choose new model settings after
seeing these test results. Stronger conclusions need independently validated labels
and recording groups, additional captures and a predeclared external evaluation.

![Pooled class trajectories](plots/pooled_class_f1.png)

## Reproduction

`python -m src.analysis.ml_results` reads saved results only and creates this new
research_analysis directory. It refuses to overwrite an existing analysis directory.
`validation.json` records hashes of protected inputs and new outputs. No retraining.
"""
    return text


def run(root=ROOT):
    root=Path(root)
    output=root/"results/ml/comparisons/research_analysis"
    if output.exists():raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    protected=[p for p in (root/"results").rglob("*") if p.is_file()]
    protected += list((root/"data/processed").rglob("*.csv"))
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    tables=prepare(root)
    output.mkdir(parents=True,exist_ok=False)
    (output/"tables").mkdir();(output/"plots").mkdir()
    for name,table in tables.items():table.to_csv(output/"tables"/f"{name}.csv",index=False,lineterminator="\n")
    plots(tables,output/"plots")
    (output/"REPORT.md").write_text(report(tables),encoding="utf-8")
    after={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    if hashes!=after:raise RuntimeError("Protected research outputs changed")
    evidence=dict(no_training=True,saved_metrics_recomputed=True,protected_files_unchanged=True,
                  protected_sha256=hashes,output_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/"validation.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(f"Created analysis: {output}; {len(tables)} tables, 5 figures; {len(hashes)} protected files unchanged")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    run(parser.parse_args().root)
