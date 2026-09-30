import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from radar import config as C
from radar import data as D
from radar import preprocessing as P
from radar.augment import StrongAug, StyleAug, build_feature_aug
from radar.backbone import RadDino
from radar.external import proxy_df
from radar.heads import make_head, update_ema
from radar.metrics import bootstrap_worst_site_f1
from radar.segmentation import load_bboxes
from radar.utils import get_device, get_logger, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--aug", action="store_true")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--sites", nargs="+", default=["montgomery", "shenzhen"])
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()

log = get_logger()
set_seed(args.seed)
dev = get_device()
epochs = args.epochs or (14 if args.aug else 10)
bs = 20 if args.aug else 24
ls = 0.05 if args.aug else 0.0

bb = RadDino(dev)
df = D.load_df(C.TRAIN_CSV)
bboxes = load_bboxes("train")
res = D.style_reservoir(df)
refs = [P.prepare_image(P.load_raw(D.image_path(r.new_id, "train")), bbox=bboxes.get(str(r.new_id)), square=True)
        for r in res.sample(min(50, len(res)), random_state=42).itertuples()] if len(res) else []
style = StyleAug(refs, 0.5)
if args.aug:
    augment = StrongAug(style, args.seed)
else:
    base_aug = build_feature_aug()

    def augment(im):
        return base_aug(style(im))
tr_df, _ = D.cr_train_split(df, 0.15, 42)


class CRData(Dataset):
    def __init__(self, sub):
        self.sub = sub.reset_index(drop=True)

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, i):
        r = self.sub.iloc[i]
        nid = str(r.new_id)
        img = P.prepare_image(P.load_raw(D.image_path(nid, "train")), bbox=bboxes.get(nid), square=True)
        if args.aug:
            return bb.preprocess(augment(img)), bb.preprocess(augment(img)), int(r.label)
        return bb.preprocess(augment(img)), int(r.label)


y = tr_df.label.to_numpy().astype(int)
w = (1.0 / np.bincount(y))[y]
sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
loader = DataLoader(CRData(tr_df), batch_size=bs, sampler=sampler, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

ext = []
for name in args.sites:
    edf = proxy_df(name)
    ebb = load_bboxes(name)
    xs = [bb.preprocess(P.prepare_image(P.load_raw(p), bbox=ebb.get(str(i)), square=True))
          for i, p in zip(edf.new_id, edf.path)]
    ext.append((name, torch.stack(xs), edf.label.to_numpy().astype(int)))

g = make_head(dev)
ema = copy.deepcopy(g)
for p in ema.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(g.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")


@torch.no_grad()
def site_probs():
    ema.eval()
    out = {}
    for name, X, yv in ext:
        ps = [torch.sigmoid(ema([bb.tokens(X[i:i + 24])])).float().cpu().numpy() for i in range(0, len(X), 24)]
        out[name] = (np.concatenate(ps), yv)
    return out


best, best_state = -1.0, None
for ep in range(epochs):
    g.train()
    for batch in loader:
        yb = batch[-1].float().to(dev)
        yt = yb * (1 - ls) + 0.5 * ls
        opt.zero_grad()
        with torch.autocast("cuda"):
            if args.aug:
                l1 = g([bb.tokens(batch[0])])
                l2 = g([bb.tokens(batch[1])])
                sup = 0.5 * (F.binary_cross_entropy_with_logits(l1, yt) + F.binary_cross_entropy_with_logits(l2, yt))
                loss = sup + F.mse_loss(torch.sigmoid(l1), torch.sigmoid(l2))
            else:
                loss = F.binary_cross_entropy_with_logits(g([bb.tokens(batch[0])]), yt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        update_ema(ema, g)
    score = bootstrap_worst_site_f1(site_probs(), n_boot=100)
    log.info("[%s seed%d ep%d] worst-site F1=%.4f", "aug" if args.aug else "plain", args.seed, ep, score)
    if score > best:
        best, best_state = score, copy.deepcopy(ema.state_dict())

C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
out = C.WEIGHTS_DIR / f"dep_{'aug' if args.aug else 'plain'}_seed{args.seed}.pt"
torch.save(best_state, out)
log.info("saved %s (worst-site F1 %.4f)", out, best)
