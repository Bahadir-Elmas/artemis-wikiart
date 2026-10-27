"""Per-movement analysis of SigLIP2 So400m/16-384 at the 20-movement level (Section VI, Table 9).

Recall, precision and F1 are computed from the confusion matrices pooled over the 10 test folds of
the artist-disjoint 10-fold cross-validation (and, for comparison, over the 10 SSS repetitions and for
CLIP ViT-L/14). The most frequent confusions and the most confused movement pairs are also listed.

Usage:  python analysis/per_class_metrics.py       (from the repository root)
Output: results/analysis/L20_SigLIP2_per_class.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
OUT = ROOT / "results" / "analysis"
P = "classification_cv_results_full_artist_disjoint_L20"
SIGLIP = CV / f"{P}_siglip2_so400m-full.xlsx"
CLIP = CV / f"{P}_clip_vitl14-full.xlsx"

# Paintings and artists per movement in the 20-movement data set (Table 2 of the paper).
PAINTINGS_ARTISTS = {
    "Action_painting": (93, 7), "Analytical_Cubism": (110, 5), "Synthetic_Cubism": (216, 7),
    "New_Realism": (313, 6), "Contemporary_Realism": (481, 15), "Pointillism": (501, 33),
    "Fauvism": (923, 72), "Minimalism": (1328, 111), "Pop_Art": (1483, 78),
    "Color_Field_Painting": (1567, 99), "Cubism": (2177, 137), "Naive_Art_Primitivism": (2385, 69),
    "Abstract_Expressionism": (2594, 140), "Art_Nouveau_Modern": (4168, 78), "Symbolism": (4524, 92),
    "Post_Impressionism": (6307, 182), "Expressionism": (6335, 237), "Romanticism": (6919, 105),
    "Realism": (10546, 176), "Impressionism": (13028, 230)}


def confusion(path, scheme):
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if s.startswith("CM_") and s.endswith("_" + scheme))
    return xl.parse(sheet, index_col=0)          # rows: true movement, columns: predicted movement


def per_class(cm):
    C = cm.values.astype(float)
    tp, support, predicted = np.diag(C), C.sum(1), C.sum(0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    pre = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros_like(tp), where=(pre + rec) > 0)
    return pd.DataFrame({"movement": cm.index, "test_paintings": support.astype(int),
                         "recall": rec, "precision": pre, "f1": f1})


if __name__ == "__main__":
    cm_k, cm_s, cm_c = confusion(SIGLIP, "kfold"), confusion(SIGLIP, "sss"), confusion(CLIP, "kfold")
    m = per_class(cm_k).merge(per_class(cm_s)[["movement", "recall", "precision", "f1"]],
                              on="movement", suffixes=("", "_sss"))
    clip = per_class(cm_c).set_index("movement")
    m["paintings"] = m["movement"].map(lambda a: PAINTINGS_ARTISTS[a][0])
    m["artists"] = m["movement"].map(lambda a: PAINTINGS_ARTISTS[a][1])
    m["clip_recall"] = m["movement"].map(clip["recall"])
    m["clip_f1"] = m["movement"].map(clip["f1"])
    m["recall_diff_siglip2_minus_clip"] = m["recall"] - m["clip_recall"]
    m["f1_diff_siglip2_minus_clip"] = m["f1"] - m["clip_f1"]
    m = m.sort_values("recall", ascending=False).reset_index(drop=True)

    C, names = cm_k.values.astype(float), list(cm_k.index)
    conf = pd.DataFrame([dict(true_movement=a, predicted_movement=b, count=int(C[i, j]),
                              share_of_true_movement=C[i, j] / C[i].sum(),
                              share_of_predicted_column=C[i, j] / C[:, j].sum())
                         for i, a in enumerate(names) for j, b in enumerate(names) if i != j and C[i, j] > 0])
    conf = conf.sort_values("share_of_true_movement", ascending=False).reset_index(drop=True)
    pairs = pd.DataFrame([dict(movement_a=names[i], movement_b=names[j], a_to_b=int(C[i, j]), b_to_a=int(C[j, i]),
                               total=int(C[i, j] + C[j, i]),
                               share_of_both_test_sets=(C[i, j] + C[j, i]) / (C[i].sum() + C[j].sum()))
                          for i in range(len(names)) for j in range(i + 1, len(names))])
    pairs = pairs.sort_values("share_of_both_test_sets", ascending=False).reset_index(drop=True)
    row_pct = pd.DataFrame(C / C.sum(1, keepdims=True), index=names, columns=names)

    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / "L20_SigLIP2_per_class.xlsx", engine="openpyxl") as w:
        m.to_excel(w, sheet_name="Per_class_metrics", index=False)
        conf.to_excel(w, sheet_name="Most_frequent_confusions", index=False)
        pairs.to_excel(w, sheet_name="Confused_pairs", index=False)
        cm_k.to_excel(w, sheet_name="CM_counts_kfold")
        row_pct.to_excel(w, sheet_name="CM_row_share_kfold")

    pd.set_option("display.width", 220)
    print(m[["movement", "test_paintings", "recall", "precision", "f1", "artists"]].round(3).to_string(index=False))
    print(f"\n10-fold pooled accuracy: {np.trace(C) / C.sum():.4f}   macro recall: {m['recall'].mean():.4f}")
    print("Written:", OUT / "L20_SigLIP2_per_class.xlsx")
