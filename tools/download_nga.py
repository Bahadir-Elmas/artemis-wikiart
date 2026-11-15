"""Downloads the open-access images of the National Gallery of Art (Washington) paintings listed in
data/external_nga/nga_paintings.csv and writes the metadata file used by the feature-extraction notebooks.

The list was built from the museum's open data (https://github.com/NationalGalleryOfArt/opendata, CC0):
paintings with exactly one style term that corresponds to one of the 20 movements, with an open-access
primary image. Images are requested through the museum's IIIF service with a long side of at most 2048 px.

Usage (from the repository root, or from a notebook with %run):
    python tools/download_nga.py            # all images; finished files are skipped
    python tools/download_nga.py --limit 20 # quick test
Set ARTEMIS_ROOT if the data folder is not under the repository folder.
"""
import os, sys, time, argparse, urllib.request, urllib.error
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("ARTEMIS_ROOT", Path(__file__).resolve().parents[1])).resolve()
EXT = ROOT / "data" / "external_nga"
IMG = EXT / "images"
SIZES = ("!2048,2048", "!1024,1024", "max", "full")   # first size the server accepts
HEADERS = {"User-Agent": "artemis-wikiart research script (academic use; one request at a time)"}


def fetch(url, dest, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            with Image.open(tmp) as im:
                im.verify()
            tmp.replace(dest)
            return True, ""
        except Exception as e:  # network error, HTTP error or unreadable image
            err = f"{type(e).__name__}: {e}"
            time.sleep(3 * (k + 1))
    return False, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args()
    lst = pd.read_csv(EXT / "nga_paintings.csv")
    if args.limit:
        lst = lst.groupby("movement", group_keys=False).head(max(1, args.limit // lst.movement.nunique() + 1)).head(args.limit)
    IMG.mkdir(parents=True, exist_ok=True)
    print(f"Project root: {ROOT}\n{len(lst)} images to download into {IMG}")
    failed, t0 = [], time.time()
    for i, r in enumerate(lst.itertuples(), 1):
        dest = IMG / f"{r.objectid}.jpg"
        if not dest.is_file():
            for size in SIZES:
                ok, err = fetch(f"{r.iiifurl}/full/{size}/0/default.jpg", dest, tries=2)
                if ok:
                    break
            if not ok:
                failed.append((r.objectid, err))
            time.sleep(args.delay)
        if i % 50 == 0 or i == len(lst):
            print(f"  {i}/{len(lst)}  ({(time.time() - t0) / 60:.1f} min, {len(failed)} failed)", flush=True)
    full = pd.read_csv(EXT / "nga_paintings.csv")
    full["filename"] = full.objectid.astype(str) + ".jpg"
    have = full[[ (IMG / f).is_file() for f in full.filename ]].copy()
    have["genre"] = ""
    have[["filename", "movement", "artist", "genre", "objectid", "title", "endyear"]].to_parquet(EXT / "nga_metadata.parquet", index=False)
    print(f"\nImages on disk: {len(have)} of {len(full)}")
    print(have.movement.value_counts().to_string())
    if failed:
        print(f"\n{len(failed)} downloads failed; run the script again to retry. First errors:")
        for o, e in failed[:5]:
            print("  ", o, e)
    else:
        print("\n✓ finished; metadata written to", EXT / "nga_metadata.parquet")


if __name__ == "__main__":
    main()
