import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from radar import config as C
from radar.heads import GradReverse, glori_input, head_logits, make_head, update_ema
from radar.metrics import auc, cross_site_worst_f1
from radar.tokens import TokenBank, load_split, split_name
from radar.utils import get_device, get_logger, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["dann", "cdan"], default="dann")
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--res", type=int, default=C.RES)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
ap.add_argument("--select", choices=["worst_f1", "mean_auc"], default=None)
ap.add_argument("--sites", nargs="+", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--preload", action="store_true")
args = ap.parse_args()

hires = args.res != C.RES
epochs = args.epochs or (10 if hires else 25)
select = args.select or ("mean_auc" if hires else "worst_f1")
sites = args.sites or (["montgomery", "shenzhen", "pakistan", "india"] if select == "mean_auc"
                       else ["montgomery", "shenzhen"])
tag = args.tag or ("hires" if hires else {"dann": "adv", "cdan": "cdan"}[args.method])

log = get_logger()
dev = get_device()
m2i = {m: i for i, m in enumerate(C.MODALITIES)}
tr = pd.read_csv(C.TRAIN_CSV, encoding="utf-8-sig")
id2mod = dict(zip(tr["new_id"].astype(str).str.strip(), tr["Modality_DICOM"].astype(str).str.strip().str.upper()))

X, ids, y = load_split(split_name("train", args.res))
Xtr = TokenBank([X], preload=args.preload)
ytr = torch.from_numpy(np.asarray(y).astype(int))
mtr = torch.tensor([m2i.get(id2mod.get(i, "CR"), 0) for i in ids]).long()
log.info("train %d images | TB %d | modality %s", len(Xtr), int((ytr == 1).sum()),
         {m: int((mtr == i).sum()) for m, i in m2i.items()})
EV = {s: load_split(split_name(s, args.res)) for s in sites}
sw = (1.0 / torch.bincount(ytr).float())[ytr]


@torch.no_grad()
def probs(head, s):
    head.eval()
    X, _, y = EV[s]
    ps = [torch.sigmoid(head_logits(head, torch.from_numpy(np.ascontiguousarray(X[k:k + 64])).to(dev)))
          .float().cpu().numpy().ravel() for k in range(0, len(X), 64)]
    return np.concatenate(ps), y


def score(head):
    if select == "mean_auc":
        return float(np.nanmean([auc(*probs(head, s)) for s in sites]))
    (pa, ya), (pb, yb) = probs(head, sites[0]), probs(head, sites[1])
    return cross_site_worst_f1(pa, ya, pb, yb)


def train_one(seed):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    g = make_head(dev)
    ema = copy.deepcopy(g)
    for p in ema.parameters():
        p.requires_grad_(False)
    d_in = C.EMBED * 2 if args.method == "cdan" else C.EMBED
    disc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, len(C.MODALITIES))).to(dev)
    opt = torch.optim.AdamW(list(g.parameters()) + list(disc.parameters()), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ls = 0.05
    steps = len(ytr) // args.bs
    best, best_state = -1.0, None
    for ep in range(epochs):
        g.train()
        idx = torch.multinomial(sw, steps * args.bs, replacement=True, generator=gen).view(steps, args.bs)
        for b in range(steps):
            bi = idx[b]
            xb = Xtr.batch(bi.numpy()).to(dev, torch.float32)
            yt = ytr[bi].to(dev, torch.float32) * (1 - ls) + 0.5 * ls
            mb = mtr[bi].to(dev)
            opt.zero_grad()
            with torch.autocast("cuda"):
                logit, feat = g(glori_input(xb), return_feat=True)
                tb = F.binary_cross_entropy_with_logits(logit.float(), yt)
                if args.method == "cdan":
                    p = torch.sigmoid(logit.float()).detach().clamp(1e-4, 1 - 1e-4)
                    gv = torch.stack([1 - p, p], dim=1)
                    T = (feat.unsqueeze(2) * gv.unsqueeze(1)).flatten(1)
                    H = -(gv * torch.log(gv)).sum(1)
                    w = (1.0 + torch.exp(-H)).detach()
                    w = w / w.mean()
                    ml = (F.cross_entropy(disc(GradReverse.apply(T, args.lam)), mb, reduction="none") * w).mean()
                else:
                    ml = F.cross_entropy(disc(GradReverse.apply(feat, args.lam)), mb)
                loss = tb + ml
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, g)
        s = score(ema)
        if s > best:
            best, best_state = s, copy.deepcopy(ema.state_dict())
        log.info("[%s seed%d ep%2d] %s=%.4f (best %.4f)", tag, seed, ep, select, s, best)
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.WEIGHTS_DIR / f"{tag}_seed{seed}.pt"
    torch.save(best_state, out)
    log.info("saved %s", out)


for s in args.seeds:
    train_one(s)
