"""External validation on the National Gallery of Art (NGA) paintings (Sections 3.9 and 4.9, Table 12,
Supplementary Section S5).

Reads the workbooks results/cv/external_nga_full_L20*_external-nga.xlsx written by notebook 03
(tools/run_external_validation.py) and computes
  - accuracy, standard deviation across the five training runs and top-3 accuracy of the ten configurations;
  - accuracy on the paintings whose artist is not among the WikiArt artists, and without the paintings that
    NGA labels Realist but that are dated 1848 or earlier;
  - paired accuracy differences with bootstrap confidence intervals (artist attributions are resampled);
  - movement-level recall and the predictions for the Realist paintings by date;
  - an estimate, from titles, of the number of NGA paintings that WikiArt also contains.

Usage:  python analysis/external_validation.py            (from the repository root)
Output: results/analysis/external_nga.xlsx and a printed summary
"""
import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
OUT = ROOT / "results" / "analysis"
N_BOOT, SEED, EARLY_YEAR = 10000, 42, 1848

# configuration: (suffix of the workbook, 10-fold accuracy on WikiArt from Table 3)
CONFIGS = {
    "ResNet50": ("", 33.1),
    "DINOv2 ViT-L/14": ("_dinov2_vitl14-full", 41.7),
    "ConvNeXt-L (IN-22k)": ("_convnext_l_in22k-full", 42.7),
    "DINOv3 ViT-L/16": ("_dinov3_vitl16-full", 45.7),
    "CLIP ViT-L/14": ("_clip_vitl14-full", 48.4),
    "ConvNeXt-L (CLIP, 256)": ("_convnext_l_clip-full", 49.6),
    "ConvNeXt-L (CLIP, 320)": ("_convnext_l_clip_320-full", 50.1),
    "SigLIP2 So400m/14 (224)": ("_siglip2_so400m_224-full", 52.1),
    "OpenCLIP ViT-H/14": ("_clip_vith14-full", 52.5),
    "SigLIP2 So400m/16 (384)": ("_siglip2_so400m-full", 53.4),
}
MAIN = "SigLIP2 So400m/16 (384)"
PAIRS = [("CLIP ViT-L/14", "DINOv2 ViT-L/14"), ("ConvNeXt-L (CLIP, 256)", "ConvNeXt-L (IN-22k)"),
         (MAIN, "CLIP ViT-L/14"), (MAIN, "OpenCLIP ViT-H/14"), ("ConvNeXt-L (CLIP, 256)", "CLIP ViT-L/14"),
         (MAIN, "SigLIP2 So400m/14 (224)"), ("ConvNeXt-L (CLIP, 320)", "ConvNeXt-L (CLIP, 256)")]
# corrections of the automatic name matching (checked by hand): a different painter with the same name,
# and a WikiArt artist listed under a different form of the name
NOT_IN_WIKIART = {"Joseph Wright"}
IN_WIKIART = {"Sir Edwin Landseer"}


def norm(s, digits=False):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+" if digits else r"[^a-z]+", " ", s).strip()


def match_artist(name, wiki):
    """WikiArt artist with the same last name whose name tokens contain, or are contained in, those of `name`."""
    t = norm(name).split()
    if not t or re.search(r"century|anonymous|unknown|school|follower|style of|after ", norm(name)):
        return None
    for a, u in wiki.items():
        if u and t[-1] == u[-1] and (set(u) <= set(t) or set(t) <= set(u)):
            return a
    return None


def main():
    P = {k: pd.read_excel(CV / f"external_nga_full_L20{s}_external-nga.xlsx", sheet_name="Predictions") for k, (s, _) in CONFIGS.items()}
    R = {k: pd.read_excel(CV / f"external_nga_full_L20{s}_external-nga.xlsx", sheet_name="Runs") for k, (s, _) in CONFIGS.items()}
    base = P[MAIN]
    for k in P:
        assert (P[k].filename.values == base.filename.values).all()
    y = base.movement.to_numpy(dtype=object)
    year = base.endyear.to_numpy(dtype=float)
    artist = base.artist.astype(str).to_numpy(dtype=object)
    n_runs = len(R[MAIN])
    pred = {k: np.array([P[k][f"pred_run{r}"].to_numpy(dtype=object) for r in range(1, n_runs + 1)]) for k in P}
    ok = {k: pred[k] == y for k in P}                       # runs x paintings

    wl = pd.read_csv(ROOT / "data" / "metadata" / "image_list.csv.gz")
    wiki = {a: norm(a).split() for a in wl.artist.unique()}
    m = {a: match_artist(a, wiki) for a in np.unique(artist)}
    for a in NOT_IN_WIKIART:
        m[a] = None
    seen = np.array([(m[a] is not None) or (a in IN_WIKIART) for a in artist])
    early = (y == "Realism") & (year <= EARLY_YEAR)

    rows = []
    for k, (_, w) in CONFIGS.items():
        acc = 100 * ok[k].mean(1)
        rows.append({"configuration": k, "wikiart_10fold_accuracy": w, "nga_accuracy": acc.mean(), "sd_runs": acc.std(ddof=1),
                     "nga_top3_accuracy": 100 * R[k].top3_accuracy.mean(), "accuracy_artists_not_in_wikiart": 100 * ok[k][:, ~seen].mean(),
                     "accuracy_artists_in_wikiart": 100 * ok[k][:, seen].mean(), "accuracy_without_early_realist": 100 * ok[k][:, ~early].mean()})
    T = pd.DataFrame(rows).set_index("configuration")
    rho, p = stats.spearmanr(T.wikiart_10fold_accuracy, T.nga_accuracy)

    ua = np.unique(artist)
    idx = {a: np.where(artist == a)[0] for a in ua}
    rng = np.random.default_rng(SEED)
    boots = [np.concatenate([idx[a] for a in rng.choice(ua, len(ua))]) for _ in range(N_BOOT)]

    def boot(v, mask=None):
        mask = np.ones(len(v), bool) if mask is None else mask
        bs = np.array([v[b][mask[b]].mean() for b in boots])
        return 100 * v[mask].mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)

    prow = []
    for a, b in PAIRS:
        d = ok[a].mean(0) - ok[b].mean(0)
        per_run = 100 * (ok[a].mean(1) - ok[b].mean(1))
        r = {"comparison": f"{a} - {b}", "runs_positive": int((per_run > 0).sum())}
        for name, mask in [("all", None), ("not_in_wikiart", ~seen), ("without_early_realist", ~early)]:
            r[f"diff_{name}"], r[f"lo_{name}"], r[f"hi_{name}"] = boot(d, mask)
        prow.append(r)
    D = pd.DataFrame(prow).set_index("comparison")
    main_ci = boot(ok[MAIN].mean(0))

    mrow = []
    for mv in pd.Series(y).value_counts().index:
        k = y == mv
        wrong = (pd.Series(pred[MAIN][:, k].ravel()).value_counts() / n_runs).drop(mv, errors="ignore").head(2)
        mrow.append({"movement": mv, "paintings": int(k.sum()), "attributions": int(pd.Series(artist[k]).nunique()),
                     "paintings_artists_not_in_wikiart": int((k & ~seen).sum()),
                     "recall_main": 100 * ok[MAIN][:, k].mean(), "recall_clip_vitl14": 100 * ok["CLIP ViT-L/14"][:, k].mean(),
                     "recall_dinov2": 100 * ok["DINOv2 ViT-L/14"][:, k].mean(),
                     "most_frequent_wrong_predictions": "; ".join(f"{c} ({100 * v / k.sum():.1f}%)" for c, v in wrong.items())})
    M = pd.DataFrame(mrow).set_index("movement")

    per = pd.cut(year, [0, 1800, EARLY_YEAR, 1880, 2100], labels=["1800 or earlier", "1801-1848", "1849-1880", "after 1880"])
    rrow = []
    for lab in per.categories:
        k = (y == "Realism") & (per == lab)
        rrow.append({"date": lab, "paintings": int(k.sum()), "predicted_realism": 100 * (pred[MAIN][:, k] == "Realism").mean(),
                     "predicted_romanticism": 100 * (pred[MAIN][:, k] == "Romanticism").mean()})
    RP = pd.DataFrame(rrow).set_index("date")

    # paintings that WikiArt may also contain: same artist and (nearly) the same title
    wl["title"] = (wl.filename.str.split("/").str[-1].str.replace(".jpg", "", regex=False).str.split("_").str[1:].str.join(" ")
                   .map(lambda s: re.sub(r"\b(1[5-9]\d\d|20\d\d)\b(?: \d)?$", "", norm(s, True)).strip()))
    titles = {a: set(g) for a, g in wl.groupby("artist").title}
    shared = 0
    for a, t in zip(artist[seen], base.title.astype(str).to_numpy(dtype=object)[seen]):
        ts = titles.get(m.get(a), set())
        t = norm(t, True)
        shared += bool(t in ts or difflib.get_close_matches(t, list(ts), 1, 0.9))

    S = pd.Series({"paintings": len(y), "attributions": len(ua), "training_runs": n_runs,
                   "paintings_artists_in_wikiart": int(seen.sum()), "artists_in_wikiart": int(pd.Series(artist[seen]).nunique()),
                   "paintings_artists_not_in_wikiart": int((~seen).sum()), "attributions_not_in_wikiart": int(pd.Series(artist[~seen]).nunique()),
                   "early_realist_paintings": int(early.sum()), "title_matches_with_wikiart": shared,
                   "spearman_wikiart_vs_nga_accuracy": rho, "spearman_p": p,
                   "main_accuracy": main_ci[0], "main_ci_low": main_ci[1], "main_ci_high": main_ci[2]}, name="value")

    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "external_nga.xlsx") as xw:
        S.to_frame().to_excel(xw, sheet_name="Summary")
        T.round(2).to_excel(xw, sheet_name="Table12")
        D.round(2).to_excel(xw, sheet_name="Differences")
        M.round(2).to_excel(xw, sheet_name="Movements")
        RP.round(2).to_excel(xw, sheet_name="Realist_by_date")
    pd.set_option("display.width", 250)
    print(S.to_string(), "\n\n", T.round(1).to_string(), "\n\n", D.round(1).to_string(), "\n\n", M.round(1).to_string(), "\n\n", RP.round(1).to_string())
    print(f"\nWritten: {OUT / 'external_nga.xlsx'}")


if __name__ == "__main__":
    main()
