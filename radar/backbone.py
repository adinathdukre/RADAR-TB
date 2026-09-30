from __future__ import annotations

import numpy as np

from . import config as C
from .utils import get_device


class RadDino:
    def __init__(self, device=None):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device or get_device()
        self.processor = AutoImageProcessor.from_pretrained(C.HF_ID)
        self.model = AutoModel.from_pretrained(C.HF_ID).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        import torch
        self.mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)

    def preprocess(self, img_u8: np.ndarray):
        from PIL import Image
        pil = Image.fromarray(img_u8).convert("RGB")
        return self.processor(images=pil, return_tensors="pt")["pixel_values"][0]

    def preprocess_res(self, img_u8: np.ndarray, res: int):
        import cv2
        import torch
        f = cv2.resize(img_u8, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = torch.from_numpy(np.repeat(f[None], 3, 0))
        return (x - self.mean[0]) / self.std[0]

    def prep(self, img_u8: np.ndarray, res: int = C.RES):
        return self.preprocess(img_u8) if res == C.RES else self.preprocess_res(img_u8, res)

    def forward(self, pix):
        o = self.model(pixel_values=pix.to(self.device), output_hidden_states=True)
        hs = o.hidden_states
        lhs = o.last_hidden_state
        cls = o.pooler_output if o.pooler_output is not None else lhs[:, 0]
        import torch
        emb = torch.cat([cls, lhs[:, self.n_prefix:].mean(dim=1)], dim=-1)
        return [(hs[i][:, self.n_prefix:], hs[i][:, 0]) for i in C.LAYERS], emb

    def tokens(self, pix):
        return self.forward(pix)[0]
