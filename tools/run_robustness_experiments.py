"""Runs the robustness experiments of the paper (Section 4.8, Table 11).

Each run executes notebooks/03_classification.ipynb with different settings and writes a new results
workbook to results/cv/ (the workbooks of the main experiments are not overwritten: the file names get a
suffix _nocw, _linear or _initNN). The 20 artist-disjoint splits are the same as in the main experiments.

Groups
  A  no class weights           BALANCE=none        all 10 feature extractors
  B  linear classifier          HEAD_HIDDEN=0       all 10 feature extractors
  C  initialisation seeds       INIT_SEED=43..46    SigLIP2 So400m/16, 10-fold scheme

CPU time: 17-43 min per run, about 10 h for the 24 runs. Runs that finished are skipped when the script is
started again.

Usage (in the TensorFlow environment used for notebook 03, from the repository root):
    python tools/run_robustness_experiments.py            # all groups
    python tools/run_robustness_experiments.py C          # only group C
    python tools/run_robustness_experiments.py A --pilot  # quick test with the pilot plan
Set ARTEMIS_ROOT if the data and features are not under the repository folder.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ARTEMIS_ROOT", Path(__file__).resolve().parents[1])).resolve()
NB = Path(__file__).resolve().parents[1] / "notebooks" / "03_classification.ipynb"
RUN_DIR = ROOT / "results" / "robustness_runs"
DONE_LOG = RUN_DIR / "done.txt"

ENCODERS = [
    "resnet50_nb02",
    "wikiart_dinov2_vitl14_full",
    "wikiart_convnext_l_in22k_full",
    "wikiart_dinov3_vitl16_full",
    "wikiart_clip_vitl14_full",
    "wikiart_convnext_l_clip_full",
    "wikiart_convnext_l_clip_320_full",
    "wikiart_siglip2_so400m_224_full",
    "wikiart_clip_vith14_full",
    "wikiart_siglip2_so400m_full",
]


def plan():
    runs = []
    for enc in ENCODERS:                       # A: no class weights
        runs.append(("A", f"A_nocw_{enc}", dict(ARTEMIS_FEATURE_SOURCE=enc, ARTEMIS_BALANCE="none")))
    for enc in ENCODERS:                       # B: linear head
        runs.append(("B", f"B_linear_{enc}", dict(ARTEMIS_FEATURE_SOURCE=enc, ARTEMIS_HEAD_HIDDEN="0")))
    for s in (43, 44, 45, 46):                 # C: initialisation seeds (seed 42 = run of the paper)
        runs.append(("C", f"C_init{s}_siglip2", dict(ARTEMIS_FEATURE_SOURCE="wikiart_siglip2_so400m_full",
                                                      ARTEMIS_INIT_SEED=str(s), ARTEMIS_SCHEMES="kfold")))
    return runs


def main():
    groups = {a for a in sys.argv[1:] if a in ("A", "B", "C")} or {"A", "B", "C"}
    pilot = "--pilot" in sys.argv
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    done = set(DONE_LOG.read_text().split()) if DONE_LOG.exists() else set()
    todo = [r for r in plan() if r[0] in groups and (pilot or r[1] not in done)]
    print(f"Repository root: {ROOT}\n{len(todo)} runs to do (groups {sorted(groups)}{', pilot' if pilot else ''})")
    for k, (_, name, env_extra) in enumerate(todo, 1):
        env = dict(os.environ, ARTEMIS_ROOT=str(ROOT), **env_extra)
        if pilot:
            env["ARTEMIS_RUN_MODE"] = "pilot"
        out_name = name + ("_pilot" if pilot else "") + ".ipynb"
        print(f"[{k}/{len(todo)}] {name} ...", flush=True)
        t0 = time.time()
        cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", str(NB),
               "--output", out_name, "--output-dir", str(RUN_DIR),
               "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3"]
        r = subprocess.run(cmd, env=env, cwd=str(NB.parent))
        mins = (time.time() - t0) / 60
        if r.returncode != 0:
            print(f"    ✗ failed after {mins:.1f} min (see {RUN_DIR / out_name} or the message above)")
            continue
        print(f"    ✓ {mins:.1f} min")
        if not pilot:
            with open(DONE_LOG, "a") as f:
                f.write(name + "\n")


if __name__ == "__main__":
    main()
