from __future__ import annotations

import os

import numpy as np

from . import config as C
from . import preprocessing as P
from .utils import get_device, get_logger, load_json, save_json

log = get_logger()
SEG_SIZE = 512


def set_xrv_cache() -> str:
    target = os.path.join(str(C.CACHE_DIR), "torchxrayvision", "models_data") + os.sep
    os.makedirs(target, exist_ok=True)
    import torchxrayvision.utils as U
    U.get_cache_dir = lambda: target
    return target


def _normalize_xrv(img_u8: np.ndarray) -> np.ndarray:
    return (2.0 * (img_u8.astype(np.float32) / 255.0) - 1.0) * 1024.0


def _largest_components(mask: np.ndarray, k: int = 2) -> np.ndarray:
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
    return np.isin(lab, [i for _, i in areas[:k]]).astype(np.uint8)


class LungSegmenter:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.model = None
        self.lung_idx: list[int] = []

    def load(self):
        set_xrv_cache()
        import torchxrayvision as xrv
        model = xrv.baseline_models.chestx_det.PSPNet().to(self.device).eval()
        targets = list(getattr(model, "targets", []))
        self.lung_idx = [i for i, t in enumerate(targets) if "lung" in str(t).lower()] or [4, 5]
        self.model = model
        return self

    def segment(self, img_u8: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        h, w = img_u8.shape[:2]
        with torch.no_grad():
            sq = img_u8 if (h, w) == (SEG_SIZE, SEG_SIZE) else cv2.resize(img_u8, (SEG_SIZE, SEG_SIZE))
            t = torch.from_numpy(_normalize_xrv(sq)[None, None]).float().to(self.device)
            prob = self.model(t).sigmoid()[0]
            m = (prob[self.lung_idx].amax(0) > 0.5).float().cpu().numpy().astype(np.uint8)
        m = _largest_components(m, k=2)
        return m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

    def bbox(self, img_u8: np.ndarray):
        bbox = P.mask_to_bbox(self.segment(img_u8))
        return bbox if bbox is not None and P.plausible_bbox(bbox, img_u8.shape) else None


def compute_bboxes(ids, paths, tag: str, segmenter: LungSegmenter | None = None,
                   margin: float = 0.08) -> dict:
    from tqdm import tqdm
    out = C.BBOX_DIR / f"{tag}.json"
    cache = load_json(out) if out.exists() else {}
    segmenter = segmenter or LungSegmenter().load()
    todo = [(str(i), p) for i, p in zip(ids, paths) if str(i) not in cache]
    for k, (nid, p) in enumerate(tqdm(todo, desc=f"segment-{tag}")):
        bbox = P.mask_to_bbox(segmenter.segment(P.load_raw(p)), margin=margin)
        cache[nid] = list(bbox) if bbox is not None else None
        if (k + 1) % 500 == 0:
            save_json(cache, out)
    save_json(cache, out)
    return load_bboxes(tag)


def load_bboxes(tag: str) -> dict:
    p = C.BBOX_DIR / f"{tag}.json"
    if not p.exists():
        return {}
    return {k: (tuple(v) if v is not None else None) for k, v in load_json(p).items()}
