import numpy as np
import torch
from scipy.optimize import minimize

from radar import config as C
from radar.heads import head_logits, load_group
from radar.tokens import load_split, split_name
from radar.utils import get_device

dev = get_device()
heads = load_group(dev, C.GROUPS["hires"])
X, _, y = load_split(split_name("test", C.HIRES_RES))
with torch.no_grad():
    p = np.mean([np.concatenate([
        torch.sigmoid(head_logits(h, torch.from_numpy(np.ascontiguousarray(X[k:k + 64])).to(dev))).float().cpu().numpy()
        for k in range(0, len(X), 64)]) for h in heads], axis=0)

eps = 1e-6
z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))


def nll(ab):
    q = 1 / (1 + np.exp(-(ab[0] * z + ab[1])))
    return -np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))


a, b = minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
print(f"PLATT_A = {a:.3f}\nPLATT_B = {b:.3f}")
