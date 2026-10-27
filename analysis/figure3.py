"""Figure 3: recall and precision of SigLIP2 So400m/16-384 per movement (20 movements, confusion matrix
pooled over the 10 test folds). Marker area grows with the number of artists; grey curves are iso-F1 lines.

Usage:  python analysis/figure3.py        (from the repository root)
Output: results/figures/figure3.pdf / .png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Times-like serif font as in the paper (TeX Gyre Termes if installed, otherwise Times New Roman / DejaVu Serif)
_have = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in ["TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "DejaVu Serif"] if f in _have), "serif")
plt.rcParams.update({"font.family": _font, "font.size": 8, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "cv" / "classification_cv_results_full_artist_disjoint_L20_siglip2_so400m-full.xlsx"
OUT = ROOT / "results" / "figures"
S = 1.2   # font scale
T = dict(x="Recall (%)", y="Precision (%)", f1="F1 = {}", few="15 or fewer artists",
         many="16 or more artists", size="Marker area: number of artists", s="{} artists")

xl = pd.ExcelFile(SRC)
cm = xl.parse(next(s for s in xl.sheet_names if s.startswith("CM_") and s.endswith("_kfold")), index_col=0)
C = cm.values.astype(float)
RECALL = dict(zip(cm.index, 100 * np.diag(C) / C.sum(1)))
PRECISION = dict(zip(cm.index, 100 * np.diag(C) / C.sum(0)))

# movement, display name, number of artists, label offset (dx, dy, ha); None = label with a leader line
D = [
    ("Romanticism", "Romanticism", 105, (-1.2, 2.6, "right")),
    ("Minimalism", "Minimalism", 111, (1.4, 2.2, "left")),
    ("Pop_Art", "Pop Art", 78, (1.6, -0.6, "left")),
    ("Color_Field_Painting", "Color Field Painting", 99, None),
    ("Abstract_Expressionism", "Abstract Expressionism", 140, (1.1, 3.4, "center")),
    ("Impressionism", "Impressionism", 230, (0.0, 3.6, "center")),
    ("Art_Nouveau_Modern", "Art Nouveau", 78, (1.4, 0.6, "left")),
    ("Realism", "Realism", 176, (-2.2, 0.0, "right")),
    ("Cubism", "Cubism", 137, (1.8, -0.8, "left")),
    ("Naive_Art_Primitivism", "Naive Art", 69, (-1.8, -1.6, "right")),
    ("Symbolism", "Symbolism", 92, (-1.8, -0.4, "right")),
    ("Synthetic_Cubism", "Synthetic Cubism", 7, (1.2, 0.0, "left")),
    ("Pointillism", "Pointillism", 33, (0.0, -3.4, "center")),
    ("Expressionism", "Expressionism", 237, (1.3, 3.8, "left")),
    ("Analytical_Cubism", "Analytical Cubism", 5, (1.2, 1.2, "left")),
    ("Contemporary_Realism", "Contemporary Realism", 15, (-1.2, -2.4, "right")),
    ("Post_Impressionism", "Post-Impressionism", 182, (0.0, -4.6, "center")),
    ("Fauvism", "Fauvism", 72, (1.6, -0.4, "left")),
    ("Action_painting", "Action painting", 7, (1.2, 0.6, "left")),
    ("New_Realism", "New Realism", 6, (1.2, -0.8, "left")),
]
D = [(name, RECALL[key], PRECISION[key], n, lab) for key, name, n, lab in D]

C_MANY, C_FEW, GRID, INK = "#2a6fbf", "#d9822b", "#c9c9c9", "#222222"
area = lambda n: 16 + 1.0 * n

fig, ax = plt.subplots(figsize=(7.16, 4.8))
ax.set_xlim(20, 90); ax.set_ylim(0, 80)

# iso-F1 curves
r = np.linspace(20.5, 90, 400)
for f in (20, 40, 60):
    p = f * r / (2 * r - f)
    m = (p > 0) & (p <= 80) & (2 * r > f)
    ax.plot(r[m], p[m], color=GRID, lw=0.7, ls=(0, (3, 2)), zorder=1)
    xl = 86.5; yl = f * xl / (2 * xl - f)
    ax.text(xl + 0.6, yl + 0.8, T["f1"].format(f), color="#8a8a8a", fontsize=7*S + 2, ha="left", va="bottom", style="italic")

for name, rec, prec, n, _ in D:
    few = n <= 15
    ax.scatter(rec, prec, s=area(n), color=C_FEW if few else C_MANY, alpha=0.85,
               edgecolor="white", linewidth=0.8, zorder=3)
for name, rec, prec, n, lab in D:
    if lab is None:   # crowded region: label with a leader line
        ax.annotate(name, (rec, prec), xytext=(rec + 1.2, prec - 6.0), fontsize=7.4*S + 2, color=INK,
                    ha="left", va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5, shrinkA=0, shrinkB=4))
        continue
    dx, dy, ha = lab
    ax.text(rec + dx, prec + dy, name, fontsize=7.4*S + 2, color=INK, ha=ha, va="center", zorder=4)

ax.set_xlabel(T["x"], fontsize=8*S + 2); ax.set_ylabel(T["y"], fontsize=8*S + 2)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=7.5*S + 2)

# legends: colour and marker size
h1 = ax.scatter([], [], s=area(60), color=C_MANY, edgecolor="white", label=T["many"])
h2 = ax.scatter([], [], s=area(6), color=C_FEW, edgecolor="white", label=T["few"])
leg1 = ax.legend(handles=[h1, h2], loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
                 fontsize=7.2*S + 2, handletextpad=0.4, borderaxespad=0.2)
ax.add_artist(leg1)
hs = [ax.scatter([], [], s=area(k), facecolor="none", edgecolor="#777777", linewidth=0.7,
                 label=T["s"].format(k)) for k in (10, 50, 200)]
ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, 0.86), frameon=False, fontsize=7.2*S + 2,
          title=T["size"], title_fontsize=7.2*S + 2*S, labelspacing=1.0, handletextpad=0.6, borderaxespad=0.2,
          alignment="left")

fig.tight_layout(pad=0.4)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "figure3.pdf"); fig.savefig(OUT / "figure3.png", dpi=600)
print("Written:", OUT / "figure3.pdf")
