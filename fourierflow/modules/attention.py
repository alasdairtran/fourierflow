from math import log, pi

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from fourierflow.common import Module
from fourierflow.utils import cache_fn, default, exists

from .rotary import SinusoidalEmbeddings, apply_rotary_emb


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, pos_emb=None):
        h = self.heads

        # Convert into the query space
        q = self.to_q(x)

        # If context doesn't exist, we simply do self attention
        context = default(context, x)

        # Convert context into the key and value space
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # The heads are independent of each other. Move them to batch dimension.
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)

        # Dot product between query and key, i.e. unnormalized attention scores
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            # mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            # mask = repeat(mask, 'b j -> (b h) () j', h=h)
            mask = repeat(mask, 'n m -> b n m', b=sim.shape[0])
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # Weighted sum of the values
        out = einsum('b i j, b j d -> b i d', attn, v)

        # Concatenate heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Final feedforward layer
        return self.to_out(out)
