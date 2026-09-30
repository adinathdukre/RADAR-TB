from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from . import config as C
from . import preprocessing as P
from .backbone import RadDino
from .heads import load_group
from .utils import get_logger, usable_device

log = get_logger()
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".dcm")
BATCH = 24


def list_images(input_dir) -> list[str]:
    root = Path(input_dir)
    if not root.is_dir():
        return [str(root)] if root.is_file() else []
    found = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not found:
        found = sorted(p for p in root.rglob("*") if p.is_file())
    return [str(p) for p in found]


def _read(path):
    try:
        return P.load_raw(path)
    except Exception as e:  # noqa: BLE001
        log.warning("could not read %s (%s); predicting from a blank frame", path, e)
        return None


class Radar:
    def __init__(self, device=None, weights_dir=None, use_crop: bool = True):
        self.device = device or usable_device()
        wd = Path(weights_dir) if weights_dir else C.WEIGHTS_DIR
        self.bb = RadDino(self.device)
        self.groups = {k: load_group(self.device, C.GROUPS[k], wd) for k in C.GROUP_WEIGHTS}
        self.hires = load_group(self.device, C.GROUPS["hires"], wd)
        for k, v in {**self.groups, "hires": self.hires}.items():
            if not v:
                raise RuntimeError(f"no weights for head group '{k}' ({C.GROUPS[k]}) under {wd}")
        ref = wd / C.POLARITY_REF
        self.ref = np.load(ref).astype(np.float32) if ref.exists() else None
        self.seg = None
        if use_crop:
            from .segmentation import LungSegmenter
            self.seg = LungSegmenter(self.device).load()
        log.info("RADAR: %s + %d hires heads on %s",
                 ", ".join(f"{k}={len(v)}" for k, v in self.groups.items()), len(self.hires), self.device)

    def _eoa(self, t, heads):
        return np.mean([torch.sigmoid(h([t])).float().cpu().numpy() for h in heads], axis=0)

    def _view(self, imgs):
        t, emb = self.bb.forward(torch.stack([self.bb.preprocess(im) for im in imgs]))
        return {k: self._eoa(t, hs) for k, hs in self.groups.items()}, emb.float().cpu().numpy()

    def _hires(self, squares):
        x = torch.stack([self.bb.preprocess_res(sq, C.HIRES_RES) for sq in squares])
        p = self._eoa(self.bb.tokens(x), self.hires)
        z = np.clip(p, 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-(C.PLATT_A * np.log(z / (1 - z)) + C.PLATT_B)))

    def _to_ref(self, e):
        return (e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-9)) @ self.ref

    def _polarity_needed(self, paths) -> bool:
        if self.ref is None:
            return False
        if len(paths) <= 96:
            return True
        idx = np.linspace(0, len(paths) - 1, 64).astype(int)
        sample = [im for im in (_read(paths[i]) for i in idx) if im is not None]
        _, ea = self._view([P.prepare_image(im, square=True) for im in sample])
        _, eb = self._view([P.prepare_image(255 - im, square=True) for im in sample])
        return bool((self._to_ref(eb) > self._to_ref(ea) + 0.02).any())

    def _crop(self, img, path):
        if self.seg is None:
            return None
        try:
            return self.seg.bbox(img)
        except Exception as e:  # noqa: BLE001
            log.warning("segmentation failed on %s (%s); using full frame", os.path.basename(path), e)
            return None

    @torch.no_grad()
    def predict_proba(self, paths) -> np.ndarray:
        polarity = self._polarity_needed(paths)
        two_view = self.seg is not None and C.W_CROP < 1.0
        probs = []
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            imgs = [im if im is not None else np.zeros((512, 512), np.uint8) for im in map(_read, chunk)]
            vfull = None
            if polarity:
                a, ea = self._view([P.prepare_image(im, square=True) for im in imgs])
                b, eb = self._view([P.prepare_image(255 - im, square=True) for im in imgs])
                flip = self._to_ref(eb) > self._to_ref(ea) + 0.02
                imgs = [255 - im if f else im for im, f in zip(imgs, flip)]
                vfull = {k: np.where(flip, b[k], a[k]) for k in a}
            elif two_view:
                vfull, _ = self._view([P.prepare_image(im, square=True) for im in imgs])
            squares = [P.prepare_image(im, bbox=self._crop(im, p), square=True) for p, im in zip(chunk, imgs)]
            v, _ = self._view(squares)
            if two_view:
                v = {k: C.W_CROP * v[k] + (1 - C.W_CROP) * vfull[k] for k in v}
            out = sum(C.GROUP_WEIGHTS[k] * v[k] for k in v)
            out = (1.0 - C.W_HIRES) * out + C.W_HIRES * self._hires(squares).reshape(np.shape(out))
            probs.append(out)
        return np.concatenate(probs) if probs else np.array([])

    def predict(self, paths, threshold: float = C.THRESHOLD):
        p = self.predict_proba(paths)
        return p, [C.INV_LABELS[int(x >= threshold)] for x in p]


def predict_dir(input_dir, out_csv, use_crop: bool = True, device=None, weights_dir=None,
                threshold: float = C.THRESHOLD, with_prob: bool = False):
    import pandas as pd
    paths = list_images(input_dir)
    log.info("found %d images under %s", len(paths), input_dir)
    probs, preds = Radar(device, weights_dir, use_crop).predict(paths, threshold)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"filename": [os.path.basename(p) for p in paths], "TB/Normal": preds})
    if with_prob:
        df["prob"] = probs
    df.to_csv(out_csv, index=False)
    log.info("wrote %s (%d TB / %d Normal)", out_csv, preds.count("TB"), preds.count("Normal"))
    return out_csv
