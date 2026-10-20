"""Prepare COCO subset for quantization + mAP evaluation (plan C).

Layout produced under data/coco:
  annotations/instances_val2017.json      (from annotations_trainval2017.zip)
  images/val2017/<file_name>              (calib + eval images, disjoint)
  manifest.json                           {"calib": [...], "eval": [...]}
and datasets/coco128 (128 train2017 images, via ultralytics) for calibration.

Split (no overlap):
  eval   = val2017 sorted by image id, first --n-eval images
  calib  = coco128 (128) + next --n-calib-extra val2017 images
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("OPENVINO_TELEMETRY_DISABLED", "1")

DATA = ROOT / "data" / "coco"
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{name}"


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dst: Path, retries: int = 3, chunk: int = 1 << 20) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total and done % (20 * chunk) < chunk:
                        log(f"  {dst.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB")
            tmp.replace(dst)
            return
        except Exception as e:  # noqa: BLE001
            log(f"  download failed ({attempt}/{retries}) {url}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}")


def ensure_annotations(force: bool = False) -> Path:
    ann = DATA / "annotations" / "instances_val2017.json"
    if ann.exists() and not force:
        log(f"[1/4] annotations already present: {ann} ({ann.stat().st_size / 1e6:.1f} MB)")
        return ann
    zip_path = DATA / "_dl" / "annotations_trainval2017.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 200_000_000:
        log("[1/4] downloading annotations_trainval2017.zip (~241 MB) ...")
        download(ANN_ZIP_URL, zip_path)
    log("[1/4] extracting instances_val2017.json ...")
    ann.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("annotations/instances_val2017.json") as src, open(ann, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink(missing_ok=True)
    log(f"[1/4] annotations ready: {ann} ({ann.stat().st_size / 1e6:.1f} MB)")
    return ann


def fetch_one(name: str, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size > 1024:
        return "skip"
    download(IMG_URL.format(name=name), dst, retries=3)
    return "ok"


def download_images(names: list[str], workers: int = 16) -> None:
    dst_dir = DATA / "images" / "val2017"
    dst_dir.mkdir(parents=True, exist_ok=True)
    todo = [(n, dst_dir / n) for n in names]
    log(f"[2/4] downloading {len(todo)} val2017 images -> {dst_dir}")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, n, p): n for n, p in todo}
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)} ({time.time() - t0:.1f}s)")
    log(f"[2/4] images ready in {time.time() - t0:.1f}s")


def ensure_coco128() -> list[Path]:
    from ultralytics import settings
    from ultralytics.data.utils import check_det_dataset

    ds_root = ROOT / "datasets"
    settings.update({"datasets_dir": str(ds_root)})
    log("[3/4] checking coco128 (ultralytics asset, ~7 MB) ...")
    data = check_det_dataset("coco128.yaml")
    imgs = sorted(Path(data["train"]).glob("*.jpg"))
    log(f"[3/4] coco128: {len(imgs)} images at {data['train']}")
    return imgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-calib-extra", type=int, default=172, help="val2017 images added to calibration")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force-annotations", action="store_true")
    ap.add_argument("--force-images", action="store_true")
    args = ap.parse_args()

    ann = ensure_annotations(force=args.force_annotations)
    with open(ann, "r", encoding="utf-8") as f:
        meta = json.load(f)
    images = sorted(meta["images"], key=lambda x: x["id"])
    n_eval = args.n_eval
    eval_imgs = images[:n_eval]
    calib_extra = images[n_eval : n_eval + args.n_calib_extra]
    log(f"[0/4] val2017 has {len(images)} images; eval={len(eval_imgs)} ids {eval_imgs[0]['id']}..{eval_imgs[-1]['id']}")
    if calib_extra:
        log(f"[0/4] calib-extra={len(calib_extra)} ids {calib_extra[0]['id']}..{calib_extra[-1]['id']}")

    if args.force_images:
        for it in eval_imgs + calib_extra:
            (DATA / "images" / "val2017" / it["file_name"]).unlink(missing_ok=True)
    download_images([it["file_name"] for it in eval_imgs + calib_extra], workers=args.workers)

    coco128 = ensure_coco128()

    manifest = {
        "eval": [{"id": it["id"], "file_name": it["file_name"]} for it in eval_imgs],
        "calib": [str(p) for p in coco128]
        + [str(DATA / "images" / "val2017" / it["file_name"]) for it in calib_extra],
        "annotations": str(ann),
    }
    mp = DATA / "manifest.json"
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[4/4] manifest: {mp}  calib={len(manifest['calib'])} eval={len(manifest['eval'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
