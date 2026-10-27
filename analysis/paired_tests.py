"""Paired split-level comparisons of the feature extractors (Section V-B, Table 5).

For every pair (A, B) and every scheme (10-fold CV, SSS) and metric, the 10 split-level differences
A - B are tested with
  - the Nadeau-Bengio corrected resampled t-test, variance factor 1/k + mean(n_test / n_train),
  - the Wilcoxon signed-rank test,
and the number of splits won by A is reported. Holm's correction is applied within the primary
comparisons, separately for every (scheme, metric) family; the remaining comparisons are exploratory
and are reported with uncorrected p-values.

Both configurations must have been evaluated on the same splits (checked through n_train).

Usage:  python analysis/paired_tests.py            (from the repository root)
Output: results/analysis/L20_paired_tests.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
OUT = ROOT / "results" / "analysis"
P = "classification_cv_results_full_artist_disjoint_L20"

FILES = {
    "ResNet50": f"{P}.xlsx",
    "CLIP ViT-L/14": f"{P}_clip_vitl14-full.xlsx",
    "OpenCLIP ViT-H/14": f"{P}_clip_vith14-full.xlsx",
    "SigLIP2 So400m/16-384": f"{P}_siglip2_so400m-full.xlsx",
    "SigLIP2 So400m/14-224": f"{P}_siglip2_so400m_224-full.xlsx",
    "DINOv2 ViT-L/14": f"{P}_dinov2_vitl14-full.xlsx",
    "DINOv3 ViT-L/16": f"{P}_dinov3_vitl16-full.xlsx",
    "ConvNeXt-L IN-22k": f"{P}_convnext_l_in22k-full.xlsx",
    "ConvNeXt-L CLIP": f"{P}_convnext_l_clip-full.xlsx",
    "ConvNeXt-L CLIP-320": f"{P}_convnext_l_clip_320-full.xlsx",
}
METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]

PAIRS = [
    ("SigLIP2 So400m/16-384", "CLIP ViT-L/14"), ("OpenCLIP ViT-H/14", "CLIP ViT-L/14"),
    ("SigLIP2 So400m/16-384", "OpenCLIP ViT-H/14"), ("SigLIP2 So400m/16-384", "SigLIP2 So400m/14-224"),
    ("SigLIP2 So400m/14-224", "CLIP ViT-L/14"), ("CLIP ViT-L/14", "DINOv2 ViT-L/14"),
    ("DINOv3 ViT-L/16", "DINOv2 ViT-L/14"), ("CLIP ViT-L/14", "DINOv3 ViT-L/16"),
    ("ConvNeXt-L IN-22k", "ResNet50"), ("ConvNeXt-L CLIP", "ConvNeXt-L IN-22k"),
    ("ConvNeXt-L CLIP", "CLIP ViT-L/14"), ("OpenCLIP ViT-H/14", "ConvNeXt-L CLIP"),
    ("ConvNeXt-L CLIP-320", "ConvNeXt-L CLIP"),
]
# Primary comparisons, tied to the research questions; Holm's correction is applied within these.
PRIMARY = {
    ("SigLIP2 So400m/16-384", "CLIP ViT-L/14"), ("SigLIP2 So400m/16-384", "OpenCLIP ViT-H/14"),
    ("CLIP ViT-L/14", "DINOv2 ViT-L/14"), ("CLIP ViT-L/14", "DINOv3 ViT-L/16"),
    ("ConvNeXt-L CLIP", "ConvNeXt-L IN-22k"), ("ConvNeXt-L CLIP", "CLIP ViT-L/14"),
    ("SigLIP2 So400m/16-384", "SigLIP2 So400m/14-224"), ("ConvNeXt-L CLIP-320", "ConvNeXt-L CLIP"),
}


def folds(name):
    d = pd.read_excel(CV / FILES[name], sheet_name="Folds")
    if d["arch"].nunique() > 1:
        raise ValueError(f"{name}: more than one configuration in the file: {d['arch'].unique()}")
    return d.sort_values(["scheme", "split"]).reset_index(drop=True)


def nb_test(a, b, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test. Returns (mean difference, two-sided p)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    corr = 1 / k + np.mean(np.asarray(n_test) / np.asarray(n_train))
    t = d.mean() / np.sqrt(corr * var) if var > 0 else np.inf
    return d.mean(), 2 * stats.t.sf(abs(t), k - 1)


def holm(p):
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, running = np.empty(m), 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def compare(A, B):
    fa, fb = folds(A), folds(B)
    rows = []
    for scheme in ["kfold", "sss"]:
        xa, xb = fa[fa.scheme == scheme], fb[fb.scheme == scheme]
        assert (xa["n_train"].values == xb["n_train"].values).all(), f"{A} vs {B}: different {scheme} splits"
        for m in METRICS:
            diff, p = nb_test(xa[m].values, xb[m].values, xa["n_train"].values, xa["n_val"].values)
            rows.append(dict(A=A, B=B, scheme=scheme, metric=m, diff_points=100 * diff, p_nb=p,
                             p_wilcoxon=stats.wilcoxon(xa[m].values, xb[m].values).pvalue,
                             splits_won_by_A=int((xa[m].values > xb[m].values).sum())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    available = {k for k, v in FILES.items() if (CV / v).exists()}
    pairs = [p for p in PAIRS if p[0] in available and p[1] in available]
    out = pd.concat([compare(a, b) for a, b in pairs], ignore_index=True)
    out["primary"] = [(a, b) in PRIMARY for a, b in zip(out.A, out.B)]
    out["p_holm"] = np.nan
    for _, g in out[out.primary].groupby(["scheme", "metric"]):
        out.loc[g.index, "p_holm"] = holm(g["p_nb"].values)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT / "L20_paired_tests.xlsx", index=False)
    pd.set_option("display.width", 200)
    print(out[out.scheme == "kfold"].round(4).to_string(index=False))
    print("\nWritten:", OUT / "L20_paired_tests.xlsx")
