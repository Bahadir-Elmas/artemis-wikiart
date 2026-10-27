"""Figure 4: confusion matrix of SigLIP2 So400m/16-384, normalised by row (20 movements, pooled over the
10 test folds). Movements are grouped by art-historical proximity; cells of 8% or more are labelled.

Usage:  python analysis/figure4.py        (from the repository root)
Output: results/figures/figure4.pdf / .png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Times-like serif font as in the paper (TeX Gyre Termes if installed, otherwise Times New Roman / DejaVu Serif)
_have = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"] if f in _have), "serif")
plt.rcParams.update({"font.family": _font, "font.size": 8, "pdf.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "cv" / "classification_cv_results_full_artist_disjoint_L20_siglip2_so400m-full.xlsx"
OUT = ROOT / "results" / "figures"
T = dict(y="True movement", x="Predicted movement", cb="Percentage of true movement")

ORDER = ["Romanticism", "Realism", "New_Realism", "Contemporary_Realism", "Impressionism", "Pointillism",
         "Post_Impressionism", "Symbolism", "Art_Nouveau_Modern", "Fauvism", "Expressionism",
         "Naive_Art_Primitivism", "Cubism", "Analytical_Cubism", "Synthetic_Cubism",
         "Abstract_Expressionism", "Action_painting", "Color_Field_Painting", "Minimalism", "Pop_Art"]
NAME = {"Art_Nouveau_Modern": "Art Nouveau", "Naive_Art_Primitivism": "Naive Art",
        "Post_Impressionism": "Post-Impressionism", "Action_painting": "Action painting"}
lab = lambda c: NAME.get(c, c.replace("_", " "))

xl = pd.ExcelFile(SRC)
cm = xl.parse(next(s for s in xl.sheet_names if s.startswith("CM_") and s.endswith("_kfold")),
             index_col=0).reindex(index=ORDER, columns=ORDER)
M = cm.values.astype(float)
R = 100 * M / M.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(7.16, 7.0))
im = ax.imshow(R, cmap="Blues", vmin=0, vmax=100, aspect="equal")
n = len(ORDER)
for i in range(n):
    for j in range(n):
        v = R[i, j]
        if v >= 8:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.8,
                    color="white" if v >= 45 else "#1a1a1a")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels([lab(c) for c in ORDER], rotation=55, ha="right", rotation_mode="anchor", fontsize=9.6)
ax.set_yticklabels([lab(c) for c in ORDER], fontsize=9.6)
ax.set_xlabel(T["x"], fontsize=10.5); ax.set_ylabel(T["y"], fontsize=10.5)
ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.6); ax.tick_params(which="minor", length=0)
ax.tick_params(length=2, width=0.5)
for s in ax.spines.values():
    s.set_linewidth(0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label(T["cb"], fontsize=9.8); cb.ax.tick_params(labelsize=9.2, width=0.5, length=2); cb.outline.set_linewidth(0.5)
fig.tight_layout(pad=0.3)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "figure4.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "figure4.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
print("Written:", OUT / "figure4.pdf")
