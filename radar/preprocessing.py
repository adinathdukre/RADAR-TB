from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _window_to_u8(img: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    flat = a[::max(1, a.shape[0] // 512), ::max(1, a.shape[1] // 512)].ravel()
    lo, hi = np.percentile(flat, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _load_dicom(path):
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    except Exception:  # noqa: BLE001
        return None
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = apply_modality_lut(ds.pixel_array, ds)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        return np.asarray(arr)
    except Exception:  # noqa: BLE001
        return None


def load_raw(path) -> np.ndarray:
    img = None
    if str(path).lower().endswith(".dcm"):
        img = _load_dicom(path)
    if img is None and cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image
        with Image.open(path) as im:
            try:
                from PIL import ImageOps
                im = ImageOps.exif_transpose(im)
            except Exception:  # noqa: BLE001
                pass
            img = np.array(im if im.mode not in ("P", "PA") else im.convert("RGB"))
    img = np.asarray(img)
    if img.ndim > 3:
        img = img.reshape(img.shape[0], img.shape[1], -1)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            img = img[..., 0]
        elif cv2 is not None and c in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if c == 4 else cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., :3].mean(axis=2) if c >= 3 else img.mean(axis=2)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        img = _window_to_u8(img)
    return np.ascontiguousarray(img)


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    return (((img - lo) / (hi - lo)) * 255.0 + 0.5).astype(np.uint8)


def mask_to_bbox(mask: np.ndarray, margin: float = 0.08):
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape
    my = int((y1 - y0 + 1) * margin)
    mx = int((x1 - x0 + 1) * margin)
    return (max(0, y0 - my), min(h, y1 + my + 1), max(0, x0 - mx), min(w, x1 + mx + 1))


def plausible_bbox(bbox, img_shape, min_area_frac: float = 0.18,
                   min_h_frac: float = 0.33, min_w_frac: float = 0.33) -> bool:
    if bbox is None:
        return False
    y0, y1, x0, x1 = bbox
    H, W = img_shape[:2]
    bh, bw = (y1 - y0), (x1 - x0)
    if bh <= 0 or bw <= 0 or H <= 0 or W <= 0:
        return False
    return (bh >= min_h_frac * H) and (bw >= min_w_frac * W) and \
           ((bh * bw) >= min_area_frac * H * W)


def crop_to_bbox(img: np.ndarray, bbox) -> np.ndarray:
    if bbox is None:
        return img
    y0, y1, x0, x1 = bbox
    out = img[y0:y1, x0:x1]
    return out if out.size else img


def square_pad(img_u8: np.ndarray) -> np.ndarray:
    h, w = img_u8.shape[:2]
    if h == w:
        return img_u8
    if cv2 is not None:
        if h > w:
            l = (h - w) // 2
            r = h - w - l
            return cv2.copyMakeBorder(img_u8, 0, 0, l, r, cv2.BORDER_REFLECT)
        t = (w - h) // 2
        b = w - h - t
        return cv2.copyMakeBorder(img_u8, t, b, 0, 0, cv2.BORDER_REFLECT)
    s = max(h, w)
    out = np.zeros((s, s), dtype=img_u8.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img_u8
    return out


def prepare_image(raw_u8: np.ndarray, bbox=None, square: bool = True) -> np.ndarray:
    img = crop_to_bbox(raw_u8, bbox)
    if square:
        img = square_pad(img)
    return img


def histogram_match_style(img_u8: np.ndarray, ref_u8: np.ndarray) -> np.ndarray:
    try:
        from skimage.exposure import match_histograms
        out = match_histograms(img_u8, ref_u8)
        return to_uint8(out)
    except Exception:
        return img_u8
