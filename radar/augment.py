from __future__ import annotations

import numpy as np

from . import preprocessing as P


def build_feature_aug():
    import albumentations as A
    tf = A.Compose([
        A.Affine(rotate=(-8, 8), shear=(-4, 4), translate_percent=(0.0, 0.04),
                 scale=(0.92, 1.08), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(70, 150), p=0.6),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    ])
    return lambda img_u8: tf(image=img_u8)["image"]


class StyleAug:
    def __init__(self, refs: list, p: float = 0.5):
        self.refs, self.p = refs, p

    def __call__(self, img_u8: np.ndarray) -> np.ndarray:
        if self.refs and np.random.rand() < self.p:
            return P.histogram_match_style(img_u8, self.refs[np.random.randint(len(self.refs))])
        return img_u8


class StrongAug:
    def __init__(self, style: StyleAug, seed: int):
        self.style = style
        self.base = build_feature_aug()
        self.rng = np.random.default_rng(seed)

    def randconv(self, u8):
        import cv2
        k = int(self.rng.choice([1, 3, 5, 7]))
        if k == 1:
            return u8
        w = self.rng.standard_normal((k, k)).astype(np.float32)
        w /= (np.abs(w).sum() + 1e-6)
        out = cv2.filter2D(u8.astype(np.float32), -1, w)
        a = float(self.rng.uniform(0.3, 1.0))
        return P.to_uint8(a * out + (1 - a) * u8.astype(np.float32))

    def amp_perturb(self, u8, eta: float = 0.5):
        import cv2
        f = np.fft.fftshift(np.fft.fft2(u8.astype(np.float32)))
        amp, pha = np.abs(f), np.angle(f)
        h, w = amp.shape
        field = self.rng.uniform(1 - eta, 1 + eta, size=(8, 8)).astype(np.float32)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        return P.to_uint8(np.real(np.fft.ifft2(np.fft.ifftshift((amp * field) * np.exp(1j * pha)))))

    def __call__(self, img):
        img = self.base(self.style(img))
        if self.rng.uniform() < 0.7:
            img = self.randconv(img)
        if self.rng.uniform() < 0.7:
            img = self.amp_perturb(img)
        return img
