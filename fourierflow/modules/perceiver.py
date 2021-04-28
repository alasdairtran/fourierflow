from math import log, pi

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from fourierflow.common import Module
from fourierflow.utils import cache_fn, default, exists

from .linear import GehringLinear
from .rotary import SinusoidalEmbeddings, apply_rotary_emb


def fourier_encode(x, max_freq, num_bands=4, base=2):
    # Our data spans over a distance of 2. If there are 100 data points,
    # the sampling frequency (i.e. mu) is 100 / 2 = 50 Hz.
    # The Nyquist frequency is 25 Hz.
    x = x.unsqueeze(-1)
    # x.shape == [n_axes, *data.shape, 1]
    device, dtype, orig_x = x.device, x.dtype, x

    # max_freq is mu in the paper.
    # Create a range between (2^0 == 1) and (2^L == mu/2)
    scales = torch.logspace(0., log(max_freq / 2) / log(base),
                            num_bands, base=base, device=device, dtype=dtype)

    # Add leading dimensions
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    # scales.shape == [1, 1, 1, n_bands] for 2D images

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    # x.shape == [n_axes, **data.shape, n_bands * 2]

    x = torch.cat((x, orig_x), dim=-1)
    # x.shape == [n_axes, **data.shape, n_bands * 2 + 1]
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(
            context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            GehringLinear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            GehringLinear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = GehringLinear(query_dim, inner_dim, bias=False)
        self.to_kv = GehringLinear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            GehringLinear(inner_dim, query_dim),
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
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # Weighted sum of the values
        out = einsum('b i j, b j d -> b i d', attn, v)

        # Concatenate heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # Final feedforward layer
        return self.to_out(out)


@Module.register('perceiver')
class Perceiver(Module):
    def __init__(self,
                 *,
                 num_freq_bands,
                 depth,
                 max_freq,
                 freq_base=2,
                 input_channels=3,
                 input_axis=2,
                 num_latents=512,
                 latent_dim=512,
                 cross_heads=1,
                 latent_heads=8,
                 cross_dim_head=64,
                 latent_dim_head=64,
                 num_classes=1000,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 weight_tie_layers=False,
                 fourier_encode_data=True,
                 self_per_cross_attn=1,
                 self_attn_rel_pos=True):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (
            input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        def get_cross_attn(): return PreNorm(latent_dim, Attention(latent_dim, input_dim,
                                                                   heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)

        def get_cross_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        def get_latent_attn(): return PreNorm(latent_dim, Attention(
            latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        def get_latent_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            GehringLinear(latent_dim, num_classes)
        )

        self.sinu_emb = None
        if self_attn_rel_pos:
            self.sinu_emb = SinusoidalEmbeddings(latent_dim_head)

    def forward(self, data, mask=None):
        # The input data can have an abitrary number of axes. The first
        # axis is the batch dimension, the last axis is the feature dimension.
        # All the middle axes are assumed to be some combination of spatial
        # and temporal dimensions.
        b, *axes, _, device = *data.shape, data.device
        assert len(axes) == self.input_axis, \
            'Input data must have the right number of axes.'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            def generate_grid(size):
                return torch.linspace(-1., 1., steps=size, device=device)
            axis_pos = list(map(generate_grid, axes))
            # axis_pos[i].shape = grid_size

            # This stores the coordiates of the data in the first dimension.
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            # pos.shape == [n_axes, *data.shape]

            enc_pos = fourier_encode(
                pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            # enc_pos.shape == [n_axes, **data.shape, n_bands * 2 + 1]
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')
        # data.shape == [batch_size, seq_len, n_features]

        # Initially, x is like our prior. We evolve this set of latents over
        # time. In each layer, the latents get to attend over the original
        # inputs, and then attend to itself.
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # x.shape == [batch_size, n_latents, latent_dim]

        # rotary embeddings for latents, if specified
        pos_emb = self.sinu_emb(x) if exists(self.sinu_emb) else None

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, pos_emb=pos_emb) + x
                x = self_ff(x) + x

        x = x.mean(dim=-2)
        return self.to_logits(x)


@Module.register('time_series_perceiver')
class TimeSeriesPerceiver(Module):
    def __init__(self,
                 *,
                 num_freq_bands,
                 depth,
                 max_freq,
                 freq_base=2,
                 input_channels=3,
                 input_axis=2,
                 num_latents=512,
                 latent_dim=512,
                 cross_heads=1,
                 latent_heads=8,
                 cross_dim_head=64,
                 latent_dim_head=64,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 weight_tie_layers=False,
                 fourier_encode_data=True,
                 self_per_cross_attn=1,
                 self_attn_rel_pos=True):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (
            input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        def get_cross_attn(): return PreNorm(latent_dim, Attention(latent_dim, input_dim,
                                                                   heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)

        def get_cross_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        def get_latent_attn(): return PreNorm(latent_dim, Attention(
            latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        def get_latent_ff(): return PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns,
            ]))

        self.out_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            GehringLinear(latent_dim, 7))

        self.sinu_emb = None
        if self_attn_rel_pos:
            self.sinu_emb = SinusoidalEmbeddings(latent_dim_head)

    def forward(self, data, mask=None):
        # data.shape == [batch_size, backcast_len]

        data = rearrange(data, 'b s -> b s 1')

        # The input data can have an abitrary number of axes. The first
        # axis is the batch dimension, the last axis is the feature dimension.
        # All the middle axes are assumed to be some combination of spatial
        # and temporal dimensions.
        b, *axes, _, device = *data.shape, data.device
        assert len(axes) == self.input_axis, \
            'Input data must have the right number of axes.'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            def generate_grid(size):
                return torch.linspace(-1., 1., steps=size, device=device)
            axis_pos = list(map(generate_grid, axes))
            # axis_pos[i].shape = grid_size

            # This stores the coordiates of the data in the first dimension.
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            # pos.shape == [n_axes, *data.shape]

            enc_pos = fourier_encode(
                pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            # enc_pos.shape == [n_axes, **data.shape, n_bands * 2 + 1]
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')
        # data.shape == [batch_size, seq_len, n_features]

        # Initially, x is like our prior. We evolve this set of latents over
        # time. In each layer, the latents get to attend over the original
        # inputs, and then attend to itself.
        x = repeat(self.latents, 'n d -> b n d', b=b)
        b, z, d = x.shape
        # x.shape == [batch_size, n_latents, latent_dim]

        # rotary embeddings for latents, if specified
        pos_emb = self.sinu_emb(x) if exists(self.sinu_emb) else None

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x, pos_emb=pos_emb) + x
                x = self_ff(x) + x
            # x.shape == [batch_size, n_latents, latent_dim]

        forecast = self.out_proj(x.mean(dim=-2))
        return forecast
