"""Batch augmentations shared by self-supervised PESTO training modes."""
import torch
import torch.nn as nn


class BatchRandomNoise(nn.Module):
    def __init__(self, min_snr: float = 0.1, max_snr: float = 2.0, p: float = 0.7):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def forward(self, x):
        bs = x.size(0)
        snr = torch.empty(bs, device=x.device).uniform_(self.min_snr, self.max_snr)
        snr[torch.rand_like(snr).le(self.p)] = 0
        noise_std = snr * x.view(bs, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(bs, -1)).view_as(x)
        return x + noise_std * torch.randn_like(x)


class BatchRandomGain(nn.Module):
    def __init__(self, min_gain: float = 0.5, max_gain: float = 1.5, p: float = 0.7):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def forward(self, x):
        bs = x.size(0)
        vol = torch.empty(bs, device=x.device).uniform_(self.min_gain, self.max_gain)
        vol[torch.rand_like(vol).le(self.p)] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(bs, -1)).view_as(x)
        return vol * x
