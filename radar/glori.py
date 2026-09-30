from __future__ import annotations

import torch
from torch import Tensor, nn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.norm1(tgt + self.dropout1(tgt))
        tgt2, _ = self.multihead_attn(tgt, memory, memory, average_attn_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        for mod in self.layers:
            tgt = mod(tgt, memory)
        return tgt


class GLoRI(nn.Module):
    def __init__(self, num_classes: int = 1, decoder_embedding: int = 768,
                 initial_num_features: int = 3072, use_n_blocks: int = 4):
        super().__init__()
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        query_embed = nn.Embedding(num_classes, decoder_embedding)
        query_embed.requires_grad_(False)
        self.decoder = Decoder(DecoderLayer(decoder_embedding), num_layers=1)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.decoder.num_classes = num_classes
        self.decoder.duplicate_pooling = nn.Parameter(torch.Tensor(num_classes, decoder_embedding, 1))
        self.decoder.duplicate_pooling_bias = nn.Parameter(torch.Tensor(num_classes))
        nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.use_n_blocks = use_n_blocks

    def forward(self, x, return_feat: bool = False):
        blocks = x[0][-self.use_n_blocks:]
        z = torch.cat([patch for patch, _ in blocks], dim=-1).float()
        mem = torch.relu(self.decoder.embed_standart(z))
        bs = mem.shape[0]
        tgt = self.decoder.query_embed.weight.unsqueeze(1).expand(-1, bs, -1)
        h = self.decoder(tgt, mem.transpose(0, 1)).transpose(0, 1)
        feat = h.reshape(bs, -1)
        w = self.decoder.duplicate_pooling
        out = torch.stack([torch.matmul(h[:, i, :], w[i]) for i in range(h.shape[1])], dim=1).flatten(1).to(h.dtype)
        logits = (out[:, :self.decoder.num_classes] + self.decoder.duplicate_pooling_bias).squeeze(-1)
        return (logits, feat) if return_feat else logits
