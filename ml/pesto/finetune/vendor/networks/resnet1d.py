"""Resnet1d encoder. Vendored from pesto-full src/models/networks/resnet1d.py.

One bug-fix vs upstream: `nn.ModuleList(*[...])` -> `nn.ModuleList([...])`.
Upstream unpacks the list with `*`, which works for `nn.Sequential` but is
incorrect for `nn.ModuleList` — luckily the default config has n_prefilt_layers=2
which produces a single-element generator that happens to survive the `*` unpack
when wrapped in `[...]` by Python's `*` semantics in this context. To be safe
we pass the list directly.
"""
from functools import partial

import torch
import torch.nn as nn


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features + out_features - 1,
            padding=out_features - 1,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.unsqueeze(-2)).squeeze(-2)


class Resnet1d(nn.Module):
    def __init__(self,
                 n_chan_input=1,
                 n_chan_layers=(20, 20, 10, 1),
                 n_prefilt_layers=1,
                 prefilt_kernel_size=15,
                 residual=False,
                 n_bins_in=216,
                 output_dim=128,
                 activation_fn: str = "leaky",
                 a_lrelu=0.3,
                 p_dropout=0.2,
                 **unused):
        # `unused` swallows checkpoint-specific kwargs like fc_margin, spiral,
        # final_norm that newer pesto-full trainings store but the inference
        # architecture doesn't use. Matches pesto/model.py behavior.
        super().__init__()

        self.hparams = dict(n_chan_input=n_chan_input,
                            n_chan_layers=n_chan_layers,
                            n_prefilt_layers=n_prefilt_layers,
                            prefilt_kernel_size=prefilt_kernel_size,
                            residual=residual,
                            n_bins_in=n_bins_in,
                            output_dim=output_dim,
                            activation_fn=activation_fn,
                            a_lrelu=a_lrelu,
                            p_dropout=p_dropout)

        if activation_fn == "relu":
            activation_layer = nn.ReLU
        elif activation_fn == "silu":
            activation_layer = nn.SiLU
        elif activation_fn == "leaky":
            activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        else:
            raise ValueError

        n_in = n_chan_input
        n_ch = list(n_chan_layers)
        if len(n_ch) < 5:
            n_ch.append(1)

        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_in, out_channels=n_ch[0],
                      kernel_size=prefilt_kernel_size, padding=prefilt_padding, stride=1),
            activation_layer(),
            nn.Dropout(p=p_dropout),
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=n_ch[0], out_channels=n_ch[0],
                          kernel_size=prefilt_kernel_size, padding=prefilt_padding, stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout),
            )
            for _ in range(n_prefilt_layers - 1)
        ])
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers) - 1):
            conv_layers.extend([
                nn.Conv1d(in_channels=n_ch[i], out_channels=n_ch[i + 1],
                          kernel_size=1, padding=0, stride=1),
                activation_layer(),
                nn.Dropout(p=p_dropout),
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)

        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        """x: (batch, channels, freq_bins)"""
        x = self.layernorm(x)
        x = self.conv1(x)
        for p in range(0, self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            if self.residual:
                x = prefilt_layer(x) + x
            else:
                x = prefilt_layer(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        y_pred = self.fc(x)
        return self.final_norm(y_pred)
