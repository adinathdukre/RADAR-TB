from __future__ import annotations

import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config as C
from .utils import get_logger

log = get_logger()

PROXIES = {
    "montgomery": "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
    "shenzhen": "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
    "pakistan": None,
    "india": None,
}
_IMG_EXT = (".png", ".jpg", ".jpeg")


def download(name: str) -> Path:
    dest = C.EXTERNAL_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    url = PROXIES.get(name)
    if url and not (dest / ".extracted").exists():
        zpath = dest / Path(url).name
        if not zpath.exists():
            log.info("downloading %s", url)
            tmp = zpath.with_suffix(zpath.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
        (dest / ".extracted").write_text("ok")
    return dest


def _norm_label(v):
    s = str(v).strip().lower()
    if s in ("1", "tb", "tuberculosis", "positive"):
        return 1
    if s in ("0", "normal", "healthy", "negative"):
        return 0
    return None


def _manifest(dest: Path) -> dict:
    out = {}
    for cp in dest.glob("**/labels.csv"):
        df = pd.read_csv(cp)
        cols = {c.lower(): c for c in df.columns}
        f = next((cols[k] for k in ("filename", "file", "path", "image") if k in cols), None)
        lab = next((cols[k] for k in ("label", "tb/normal", "class") if k in cols), None)
        if f and lab:
            for a, b in zip(df[f].astype(str), df[lab]):
                if _norm_label(b) is not None:
                    out[Path(a).stem] = _norm_label(b)
    return out


def _folder_label(path: str):
    s = str(path).replace("\\", "/").lower()
    if "/normal/" in s or "/health/" in s or "/healthy/" in s:
        return 0
    if "/tb/" in s or "/tuberculosis/" in s:
        return 1
    return None


def proxy_df(name: str) -> pd.DataFrame:
    dest = download(name)
    nlm = name in ("montgomery", "shenzhen")
    if nlm:
        imgs = sorted(glob.glob(str(dest / "**" / "CXR_png" / "*.png"), recursive=True))
    else:
        imgs = sorted(p for e in _IMG_EXT for p in glob.glob(str(dest / "**" / f"*{e}"), recursive=True)
                      if "mask" not in p.lower())
    manifest = {} if nlm else _manifest(dest)
    rows = []
    for i, p in enumerate(imgs):
        stem = Path(p).stem
        lab = int(stem.split("_")[-1]) if nlm else manifest.get(stem, _folder_label(p))
        if lab is None:
            continue
        rows.append({"new_id": stem if nlm else f"{name}_{i:05d}_{lab}", "path": p, "label": int(lab)})
    df = pd.DataFrame(rows, columns=["new_id", "path", "label"])
    if len(df) == 0 or df["label"].nunique() < 2:
        log.warning("proxy %s: no usable labelled images under %s", name, dest)
        return df.iloc[0:0]
    log.info("proxy %s: %d images (%d TB / %d Normal)", name, len(df),
             int((df.label == 1).sum()), int((df.label == 0).sum()))
    return df
