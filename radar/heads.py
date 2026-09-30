from __future__ import annotations

import glob

import torch

from . import config as C
from .glori import GLoRI


def make_head(device=None) -> GLoRI:
    g = GLoRI(num_classes=1, decoder_embedding=C.EMBED,
              initial_num_features=C.EMBED * len(C.LAYERS), use_n_blocks=len(C.LAYERS))
    return g.to(device) if device is not None else g


def glori_input(concat: torch.Tensor):
    chunks = concat.float().split(C.EMBED, dim=-1)
    dummy = concat.new_zeros(concat.shape[0], C.EMBED, dtype=torch.float32)
    return [[(c, dummy) for c in chunks]]


def head_logits(head, concat: torch.Tensor):
    return head(glori_input(concat))


def load_group(device, patterns, weights_dir=None):
    root = weights_dir or C.WEIGHTS_DIR
    paths = sorted(sum([glob.glob(str(root / p)) for p in patterns], []))
    heads = []
    for p in paths:
        g = make_head(device)
        g.load_state_dict(torch.load(p, map_location=device))
        heads.append(g.eval())
    return heads


def update_ema(ema, model, decay: float = 0.99) -> None:
    with torch.no_grad():
        for a, b in zip(ema.parameters(), model.parameters()):
            a.mul_(decay).add_(b, alpha=1 - decay)
        for a, b in zip(ema.buffers(), model.buffers()):
            a.copy_(b)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None
