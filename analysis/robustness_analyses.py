"""Additional analyses of the paper that are computed from the results workbooks in results/cv/.

1. 95% confidence intervals of the paired accuracy differences (Table 6), with the Nadeau-Bengio
   corrected standard error, for the 10-fold and SSS schemes.
2. Leave-one-fold-out repetition of the main-model selection (Section 3.8).
3. Analyses of the pooled confusion matrix of the main model (Sections 5.1 and 5.2): label-tolerant
   accuracy, mapping to 16 and 17 classes, effect of the four smallest movements.
4. Random versus artist-disjoint split on the pooled predictions (Section 4.5).
5. Robustness experiments (Section 4.8, Table 11): no class weights, linear classifier,
   initialisation seeds; the primary comparisons are repeated for each classifier configuration.

Usage:  python analysis/robustness_analyses.py            (from the repository root)
Output: results/analysis/L20_robustness.xlsx and a printed summary
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import paired_tests as pt

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
OUT = ROOT / "results" / "analysis"
MAIN = "SigLIP2 So400m/16-384"
VARIANTS = {"main": "", "no class weights": "_nocw", "linear": "_linear"}
PRIMARY = [p for p in pt.PAIRS if p in pt.PRIMARY]


def folds(name, suffix=""):
    f = pt.FILES[name].replace(".xlsx", f"{suffix}.xlsx")
    return pd.read_excel(CV / f, sheet_name="Folds")


def pooled_cm(file):
    xl = pd.ExcelFile(CV / file)
    sheet = [s for s in xl.sheet_names if s.startswith("CM_") and s.endswith("kfold")][0]
    return xl.parse(sheet, index_col=0)


def nb(a, b, col="accuracy"):
    """Mean difference a - b (points), 95% CI, p-value and number of splits won by a."""
    assert (a.n_train.values == b.n_train.values).all(), "the two runs do not use the same splits"
    d = 100 * (a[col].values - b[col].values)
    k = len(d)
    se = np.sqrt((1 / k + np.mean(a.n_val.values / a.n_train.values)) * d.var(ddof=1))
    half = stats.t.ppf(0.975, k - 1) * se
    p = 2 * stats.t.sf(abs(d.mean() / se), k - 1)
    return d.mean(), d.mean() - half, d.mean() + half, p, int((d > 0).sum())


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def confidence_intervals():
    rows = []
    for a, b in pt.PAIRS:
        row = {"A": a, "B": b, "primary": (a, b) in pt.PRIMARY}
        for s in ("kfold", "sss"):
            m = nb(folds(a).query("scheme == @s"), folds(b).query("scheme == @s"))
            row.update({f"{s}_diff": m[0], f"{s}_ci_low": m[1], f"{s}_ci_high": m[2], f"{s}_p": m[3]})
        rows.append(row)
    return pd.DataFrame(rows)


def selection_check():
    rows = []
    for s in ("kfold", "sss"):
        names = list(pt.FILES)
        x = np.array([100 * folds(n).query("scheme == @s").accuracy.values for n in names])
        k = x.shape[1]
        picks, values = [], []
        for f in range(k):
            others = [j for j in range(k) if j != f]
            best = x[:, others].mean(1).argmax()
            picks.append(names[best])
            values.append(x[best, f])
        rows.append({"scheme": s, "best mean": names[x.mean(1).argmax()], "best mean accuracy": x.mean(1).max(),
                     "leave-one-fold-out accuracy": np.mean(values),
                     "times main model selected": picks.count(MAIN),
                     "splits in which main model is best": int((x.argmax(0) == names.index(MAIN)).sum())})
    return pd.DataFrame(rows)


def confusion_analyses():
    cm = pooled_cm(pt.FILES[MAIN])
    names, c = list(cm.index), cm.values.astype(float)

    def acc_bal(m):
        return 100 * np.trace(m) / m.sum(), 100 * np.mean(np.diag(m) / m.sum(1))

    def mapped(mapping):
        groups = [mapping.get(n, n) for n in names]
        u = sorted(set(groups))
        m = np.zeros((len(u), len(u)))
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                m[u.index(gi), u.index(gj)] += c[i, j]
        return m

    l16 = {"Analytical_Cubism": "Cubism", "Synthetic_Cubism": "Cubism",
           "Action_painting": "Abstract_Expressionism", "New_Realism": "Contemporary_Realism"}
    l17 = {k: v for k, v in l16.items() if k != "New_Realism"}
    pairs = [("Impressionism", "Realism"), ("Fauvism", "Expressionism"), ("Symbolism", "Art_Nouveau_Modern"),
             ("Symbolism", "Expressionism"), ("Color_Field_Painting", "Abstract_Expressionism")]
    tolerant = np.trace(c) + sum(c[names.index(a), names.index(b)] + c[names.index(b), names.index(a)]
                                 for a, b in pairs)
    small = ["Analytical_Cubism", "New_Realism", "Action_painting", "Synthetic_Cubism"]
    big = [i for i, n in enumerate(names) if n not in small]
    recall = np.diag(c) / c.sum(1)
    rows = [("L20, pooled", *acc_bal(c)), ("mapped to L16", *acc_bal(mapped(l16))),
            ("mapped to 17 classes (New Realism kept separate)", *acc_bal(mapped(l17))),
            ("five co-labelled pairs counted as correct", 100 * tolerant / c.sum(), np.nan),
            ("without the four movements with fewer than ten artists",
             100 * np.diag(c)[big].sum() / c[big].sum(), 100 * recall[big].mean())]
    return pd.DataFrame(rows, columns=["analysis", "accuracy", "balanced accuracy"])


def leakage_pooled():
    rows = []
    for label, tag in (("CLIP ViT-L/14", "_clip_vitl14-full"), ("ResNet50", "")):
        a = pooled_cm(f"classification_cv_results_full_artist_disjoint_L20{tag}.xlsx").values.astype(float)
        r = pooled_cm(f"classification_cv_results_full_random_L20{tag}.xlsx").values.astype(float)
        assert (a.sum(1) == r.sum(1)).all(), "the pooled test sets differ"
        ra, rr = np.diag(a) / a.sum(1), np.diag(r) / r.sum(1)
        rows.append({"model": label, "artist-disjoint": 100 * np.trace(a) / a.sum(),
                     "random": 100 * np.trace(r) / r.sum(),
                     "difference": 100 * (np.trace(r) - np.trace(a)) / a.sum(),
                     "movements with higher recall under the random split": int((rr > ra).sum())})
    return pd.DataFrame(rows)


def classifier_configurations():
    table, effects, comparisons = [], [], []
    for name in pt.FILES:
        row = {"model": name}
        for label, suffix in VARIANTS.items():
            q = folds(name, suffix).query("scheme == 'kfold'")
            row[f"{label}: accuracy"] = 100 * q.accuracy.mean()
            row[f"{label}: balanced accuracy"] = 100 * q.balanced_accuracy.mean()
            if suffix:
                for s in ("kfold", "sss"):
                    a, b = folds(name, suffix).query("scheme == @s"), folds(name).query("scheme == @s")
                    m = nb(a, b)
                    effects.append({"model": name, "configuration": label, "scheme": s, "accuracy diff": m[0],
                                    "ci_low": m[1], "ci_high": m[2], "p": m[3],
                                    "balanced accuracy diff": nb(a, b, "balanced_accuracy")[0]})
        table.append(row)
    for label, suffix in VARIANTS.items():
        for s in ("kfold", "sss"):
            res = [nb(folds(a, suffix).query("scheme == @s"), folds(b, suffix).query("scheme == @s"))
                   for a, b in PRIMARY]
            for (a, b), m, ph in zip(PRIMARY, res, holm([m[3] for m in res])):
                comparisons.append({"configuration": label, "scheme": s, "A": a, "B": b, "diff": m[0],
                                    "ci_low": m[1], "ci_high": m[2], "p": m[3], "p_holm": ph})
    return pd.DataFrame(table), pd.DataFrame(effects), pd.DataFrame(comparisons)


def initialisation_seeds():
    runs = {42: folds(MAIN).query("scheme == 'kfold'")}
    for seed in (43, 44, 45, 46):
        runs[seed] = folds(MAIN, f"_init{seed}").query("scheme == 'kfold'")
        assert (runs[seed].n_val.values == runs[42].n_val.values).all()
    acc = np.array([100 * r.accuracy.values for r in runs.values()])
    per_seed = pd.DataFrame({"seed": list(runs), "accuracy": acc.mean(1),
                             "balanced accuracy": [100 * r.balanced_accuracy.mean() for r in runs.values()],
                             "SD across folds": acc.std(1, ddof=1)})
    summary = pd.DataFrame([{"SD of the 10-fold mean across seeds": acc.mean(1).std(ddof=1),
                             "range of the 10-fold mean": acc.mean(1).max() - acc.mean(1).min(),
                             "pooled SD across seeds within a fold": np.sqrt(acc.var(0, ddof=1).mean()),
                             "SD of the fold means": acc.mean(0).std(ddof=1)}])
    return per_seed, summary


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    table, effects, comparisons = classifier_configurations()
    per_seed, seed_summary = initialisation_seeds()
    sheets = {"CI_paired_differences": confidence_intervals(), "Selection_check": selection_check(),
              "Confusion_analyses": confusion_analyses(), "Leakage_pooled": leakage_pooled(),
              "Classifier_configurations": table, "Configuration_effects": effects,
              "Primary_comparisons": comparisons, "Seeds": per_seed, "Seeds_summary": seed_summary}
    with pd.ExcelWriter(OUT / "L20_robustness.xlsx") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for name, frame in sheets.items():
        print(f"\n== {name} ==")
        print(frame.round(2).to_string(index=False))
    print("\nWritten:", OUT / "L20_robustness.xlsx")
