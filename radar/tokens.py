from __future__ import annotations

import numpy as np
import torch

from . import config as C
from . import preprocessing as P
from .utils import get_logger

log = get_logger()


def split_name(split: str, res: int) -> str:
    return split if res == C.RES else f"{split}_r{res}"


def tok_paths(name: str):
    return C.TOKEN_DIR / f"{name}.tok.npy", C.TOKEN_DIR / f"{name}.meta.npz"


@torch.no_grad()
def cache_tokens(bb, split: str, ids, paths, bboxes: dict, labels, res: int = C.RES,
                 batch: int = 16, overwrite: bool = False):
    name = split_name(split, res)
    tp, mp = tok_paths(name)
    if tp.exists() and mp.exists() and not overwrite:
        log.info("tokens for %s already cached", name)
        return tp
    C.TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    g = res // 14
    toks = np.lib.format.open_memmap(str(tp), mode="w+", dtype=np.float16,
                                     shape=(len(ids), g * g, C.EMBED * len(C.LAYERS)))
    for k in range(0, len(ids), batch):
        xs = [bb.prep(P.prepare_image(P.load_raw(p), bbox=(bboxes or {}).get(str(i)), square=True), res)
              for i, p in zip(ids[k:k + batch], paths[k:k + batch])]
        t = bb.tokens(torch.stack(xs))
        toks[k:k + len(xs)] = torch.cat([a for a, _ in t], dim=-1).half().cpu().numpy()
        if (k // batch) % 50 == 0:
            log.info("  %s %d/%d", name, k, len(ids))
    toks.flush()
    np.savez(mp, new_id=np.array([str(i) for i in ids]), label=np.asarray(labels).astype(np.int8))
    log.info("cached %s %s", name, toks.shape)
    return tp


def load_split(name: str):
    tp, mp = tok_paths(name)
    meta = np.load(mp, allow_pickle=True)
    return np.load(tp, mmap_mode="r"), [str(i) for i in meta["new_id"]], meta["label"].astype(int)


class TokenBank:
    def __init__(self, arrays, preload: bool = False):
        if preload:
            arrays = [np.ascontiguousarray(a) for a in arrays]
        self.arrays = arrays
        self.offs = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.offs[-1])

    def batch(self, gidx) -> torch.Tensor:
        gidx = np.asarray(gidx)
        a0 = self.arrays[0]
        out = np.empty((len(gidx), *a0.shape[1:]), dtype=a0.dtype)
        for j, gi in enumerate(gidx):
            p = int(np.searchsorted(self.offs, gi, side="right") - 1)
            out[j] = self.arrays[p][gi - self.offs[p]]
        return torch.from_numpy(out)
