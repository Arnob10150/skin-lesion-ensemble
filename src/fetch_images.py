"""Fetch dermoscopy thumbnails from the ISIC S3 bucket for both HAM10000 and
ISIC 2019 (IDs already known from the ground-truth CSVs). This avoids both the
3 GB HuggingFace parquet download and the 9.7 GB challenge zip.

Thumbnails are 256-px JPEGs (~5-10 KB each); 35k images is ~300 MB.
"""
import os, io, sys, time, urllib.request
import concurrent.futures as cf
from PIL import Image
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAM_META = os.path.join(ROOT, "data", "ham10000", "raw", "HAM10000_metadata.csv")
ISIC_GT  = os.path.join(ROOT, "data", "isic2019", "raw", "ISIC_2019_Training_GroundTruth.csv")

HAM_OUT  = os.path.join(ROOT, "data", "ham10000", "images")
ISIC_OUT = os.path.join(ROOT, "data", "isic2019", "images")
os.makedirs(HAM_OUT, exist_ok=True); os.makedirs(ISIC_OUT, exist_ok=True)

# ISIC S3 thumbnail pattern: https://isic-archive.s3.amazonaws.com/thumbnails/<ISIC_ID>_thumbnail.jpg
# Some IDs don't have thumbnails (private); fall back to full image.
def _url_thumb(iid): return f"https://isic-archive.s3.amazonaws.com/thumbnails/{iid}_thumbnail.jpg"
def _url_full(iid):  return f"https://isic-archive.s3.amazonaws.com/images/{iid}.jpg"

def _fetch(args):
    iid, out = args
    if os.path.exists(out): return 1
    for url in (_url_thumb(iid), _url_full(iid)):
        for _ in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"research-client"})
                b = urllib.request.urlopen(req, timeout=20).read()
                if len(b) < 1000: continue
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img.thumbnail((320, 320), Image.BILINEAR); img.save(out, quality=92)
                return 1
            except Exception: pass
    return 0

def bulk(ids, outdir, label):
    t0 = time.time()
    jobs = [(iid, os.path.join(outdir, f"{iid}.jpg")) for iid in ids]
    done = 0; last = 0
    with cf.ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(_fetch, jobs), 1):
            done += r
            if i - last >= 500:
                dt = time.time() - t0
                print(f"[{label}] {i}/{len(jobs)} done={done}, rate {i/dt:.0f}/s, {dt:.0f}s", flush=True)
                last = i
    print(f"[{label}] finished: {done}/{len(jobs)} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    tasks = sys.argv[1:] or ["ham", "isic"]

    if "ham" in tasks:
        ham = pd.read_csv(HAM_META)
        ids = ham["image_id"].astype(str).unique().tolist()
        print(f"HAM10000: {len(ids)} image IDs")
        bulk(ids, HAM_OUT, "HAM")

    if "isic" in tasks:
        isic = pd.read_csv(ISIC_GT)
        ids = isic["image"].astype(str).unique().tolist()
        print(f"ISIC 2019: {len(ids)} image IDs")
        bulk(ids, ISIC_OUT, "ISIC")
