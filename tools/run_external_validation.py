"""External validation on another collection (Section 4.9 of the paper).

Step "features": notebook 02B extracts the features of the external images with the nine PyTorch
                 configurations (GPU kernel).
Step "classify": notebook 03 trains the classifier on all WikiArt paintings and applies it to the
                 external collection, for the ten configurations (TensorFlow kernel; ResNet50 features
                 are computed in this step).

Usage (from the repository root, or from a notebook with %run):
    python tools/run_external_validation.py features --pilot   # 20 images, one model: tests the GPU kernel
    python tools/run_external_validation.py features
    python tools/run_external_validation.py classify --pilot   # quick test with one model
    python tools/run_external_validation.py classify
Finished runs are skipped. ARTEMIS_GPU_KERNEL sets the name of the Jupyter kernel with PyTorch
(default: the first kernel whose name contains "gpu"). Set ARTEMIS_ROOT if the data are elsewhere.
"""
import os, sys, time, subprocess, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ARTEMIS_ROOT", REPO)).resolve()
OUT = ROOT / "results" / "external_runs"
GPU_MODELS = ["siglip2_so400m", "clip_vitl14", "dinov2_vitl14", "convnext_l_clip", "convnext_l_in22k",
              "dinov3_vitl16", "clip_vith14", "siglip2_so400m_224", "convnext_l_clip_320"]
SOURCES = [f"wikiart_{m}_full" for m in GPU_MODELS] + ["resnet50_nb02"]


def gpu_kernel():
    if os.environ.get("ARTEMIS_GPU_KERNEL"):
        return os.environ["ARTEMIS_GPU_KERNEL"]
    from jupyter_client.kernelspec import KernelSpecManager
    specs = KernelSpecManager().get_all_specs()
    for name, s in specs.items():
        if "gpu" in name.lower() or "gpu" in s["spec"].get("display_name", "").lower():
            return name
    raise SystemExit(f"No Jupyter kernel with 'gpu' in its name was found ({list(specs)}). "
                     f"Set ARTEMIS_GPU_KERNEL to the kernel that has PyTorch.")


def run(notebook, label, env, kernel):
    out = OUT / f"{label}.ipynb"
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", str(REPO / "notebooks" / notebook),
           "--output", str(out), "--ExecutePreprocessor.timeout=-1", f"--ExecutePreprocessor.kernel_name={kernel}"]
    t0 = time.time()
    r = subprocess.run(cmd, env=dict(os.environ, ARTEMIS_ROOT=str(ROOT), **env), cwd=str(REPO / "notebooks"),
                       capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"    {'✓' if ok else '✗'} {(time.time() - t0) / 60:.1f} min", flush=True)
    if not ok:
        print("    " + "\n    ".join(r.stderr.strip().splitlines()[-12:]), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["features", "classify"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dataset", default="nga")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done_file = OUT / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    print(f"Project root: {ROOT}")
    if a.step == "features":
        kernel = gpu_kernel()
        print(f"GPU kernel: {kernel}")
        models = GPU_MODELS[:1] if a.pilot else GPU_MODELS
        for i, m in enumerate(models, 1):
            label = f"features_{a.dataset}_{m}" + ("_pilot" if a.pilot else "")
            print(f"[{i}/{len(models)}] {label} ...", flush=True)
            if label in done and not a.pilot:
                print("    (already done)"); continue
            env = dict(ARTEMIS_DATASET=a.dataset, ARTEMIS_MODEL=m, **({"ARTEMIS_PILOT_N": "20"} if a.pilot else {}))
            if run("02B_gpu_features.ipynb", label, env, kernel) and not a.pilot:
                with open(done_file, "a") as f: f.write(label + "\n")
        if a.pilot:      # the same notebook on 50 WikiArt images must reproduce the stored WikiArt features
            m = GPU_MODELS[0]
            print(f"[check] 50 WikiArt images with {m}: comparison with the stored features ...", flush=True)
            if run("02B_gpu_features.ipynb", f"features_wikiart_{m}_pilot", dict(ARTEMIS_DATASET="wikiart", ARTEMIS_MODEL=m, ARTEMIS_PILOT_N="50"), kernel):
                import numpy as np, pandas as pd
                g = ROOT / "cache" / "embeddings" / "gpu"
                new, ni = np.load(g / f"wikiart_{m}_full_pilot50.npy"), pd.read_parquet(g / f"wikiart_{m}_full_pilot50_index.parquet")
                old, oi = np.load(g / f"wikiart_{m}_full.npy", mmap_mode="r"), pd.read_parquet(g / f"wikiart_{m}_full_index.parquet")
                rows = pd.Series(oi["embedding_index"].to_numpy(), index=oi["filename"]).reindex(ni["filename"])
                keep = rows.notna().to_numpy() & ni["load_ok"].to_numpy()
                x, y = new[keep, 0].astype(np.float32), np.asarray(old[rows[keep].astype(int).to_numpy(), 0]).astype(np.float32)
                cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
                print(f"    cosine with the stored WikiArt features: min {cos.min():.5f}, mean {cos.mean():.5f} "
                      f"({'✓ same extraction' if cos.min() > 0.995 else '✗ DIFFERENT extraction: stop and report'})")
    else:
        sources = SOURCES[:1] if a.pilot else SOURCES
        for i, s in enumerate(sources, 1):
            label = f"classify_{a.dataset}_{s}" + ("_pilot" if a.pilot else "")
            print(f"[{i}/{len(sources)}] {label} ...", flush=True)
            if label in done and not a.pilot:
                print("    (already done)"); continue
            env = dict(ARTEMIS_EXTERNAL_TEST=a.dataset, ARTEMIS_FEATURE_SOURCE=s,
                       ARTEMIS_RUN_MODE="pilot" if a.pilot else "full")
            if run("03_classification.ipynb", label, env, "python3") and not a.pilot:
                with open(done_file, "a") as f: f.write(label + "\n")


if __name__ == "__main__":
    main()
