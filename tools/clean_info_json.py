"""Rewrite the *_info.json files written by older runs of notebook 02B in the form of the current notebook:
the resize descriptions are given in English and the GPU device name is removed. The feature values,
shapes and all other fields are left unchanged. Files written before quadrant crops existed get
crop_mode = "grid_square", the only crop mode at that time.

Usage:  python tools/clean_info_json.py <folder with *_info.json> <output folder>
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in sorted(src.glob("*_info.json")):
    info = json.loads(f.read_text(encoding="utf-8"))
    if "tag" not in info:                # not a notebook-02B file (e.g. embedding_info.json of notebook 02)
        continue
    s = info["input_size"]
    info.pop("device", None)
    if "crop_mode" not in info:          # files written before quadrant crops were added: square windows only
        keys = list(info)
        info = {**{k: info[k] for k in keys[:keys.index("crop_frac")]}, "crop_mode": "grid_square",
                **{k: info[k] for k in keys[keys.index("crop_frac"):]}}
    info["full_resize"] = f"squash to {s}x{s} without preserving the aspect ratio (bicubic)"
    info["crop_resize"] = (f"square window -> {s}x{s} (bicubic)" if info.get("crop_mode") == "grid_square"
                           else f"rectangular quadrant -> {s}x{s} (bicubic, aspect ratio not preserved)")
    out = dst / f.name
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    left = [k for k, v in info.items() if isinstance(v, str) and any(c in v for c in "çğıöşüÇĞİÖŞÜ")]
    print(f"{f.name:60s} ok" + (f"   CHECK: non-English text in {left}" if left else ""))
