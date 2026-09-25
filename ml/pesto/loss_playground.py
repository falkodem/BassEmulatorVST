#!/usr/bin/env python3
"""Small playground for PESTO-style self-supervised losses.

Run from repo root:
    source venv/bin/activate
    python ml/pesto/loss_playground.py
    python ml/pesto/loss_playground.py --bins 219 --shift 8 --sharpness 3 12 40

What this shows:
- pesto CE: what pesto-full uses: torch.nn.CrossEntropyLoss(input, target),
  where input is already a softmax probability vector from the encoder.
- prob CE: textbook cross entropy between probability distributions:
  -sum(target * log(input)).

The current fine-tune code vendors the same loss semantics as pesto-full.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


EPS = torch.finfo(torch.float32).eps


def gaussian_probs(bins: int, center: float, sharpness: float) -> torch.Tensor:
    """Return a normalized bell-shaped distribution over pitch bins."""
    x = torch.arange(bins, dtype=torch.float32)
    logits = -0.5 * ((x - center) / sharpness) ** 2
    return torch.softmax(logits, dim=-1)


def one_hot(bins: int, index: int) -> torch.Tensor:
    out = torch.zeros(bins, dtype=torch.float32)
    out[index] = 1.0
    return out


def shift_distribution(probs: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift without wrap; mass shifted outside the range is dropped."""
    out = torch.zeros_like(probs)
    if shift >= 0:
        out[shift:] = probs[:-shift] if shift else probs
    else:
        out[:shift] = probs[-shift:]
    total = out.sum()
    return out / total.clamp_min(EPS)


def pesto_ce(input_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """PESTO/upstream-style CE: CrossEntropyLoss receives probabilities as input."""
    return F.cross_entropy(input_probs.unsqueeze(0), target_probs.unsqueeze(0))


def pesto_symmetric_ce(input_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return 0.5 * (pesto_ce(input_probs, target_probs) + pesto_ce(target_probs, input_probs))


def prob_ce(input_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Textbook CE H(target, input) for probability distributions."""
    return -(target_probs * input_probs.clamp_min(EPS).log()).sum()


def prob_symmetric_ce(input_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return 0.5 * (prob_ce(input_probs, target_probs) + prob_ce(target_probs, input_probs))


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return prob_ce(probs, probs)


def shift_cross_entropy_pesto(x1: torch.Tensor, x2: torch.Tensor, shift: int, pad_length: int) -> torch.Tensor:
    """Same mechanics as ml.pesto.finetune.vendor.losses.ShiftCrossEntropy."""
    x1_batch = x1.unsqueeze(0)
    x2_batch = x2.unsqueeze(0)
    target = torch.tensor([shift], dtype=torch.long)

    x1_padded = F.pad(x1_batch, (pad_length, pad_length))
    x2_padded = F.pad(x2_batch, (2 * pad_length, 2 * pad_length))
    idx = target.unsqueeze(1) + torch.arange(x1_padded.size(-1)) + pad_length
    shifted_x2 = torch.gather(x2_padded, dim=1, index=idx)
    return F.cross_entropy(x1_padded, shifted_x2)


def shift_cross_entropy_prob(x1: torch.Tensor, x2: torch.Tensor, shift: int, pad_length: int) -> torch.Tensor:
    """Probability-CE variant with the same padding/gather mechanics."""
    x1_batch = x1.unsqueeze(0)
    x2_batch = x2.unsqueeze(0)
    target = torch.tensor([shift], dtype=torch.long)

    x1_padded = F.pad(x1_batch, (pad_length, pad_length))
    x2_padded = F.pad(x2_batch, (2 * pad_length, 2 * pad_length))
    idx = target.unsqueeze(1) + torch.arange(x1_padded.size(-1)) + pad_length
    shifted_x2 = torch.gather(x2_padded, dim=1, index=idx)
    return prob_ce(x1_padded.squeeze(0), shifted_x2.squeeze(0))


@dataclass(frozen=True)
class Pair:
    name: str
    input_probs: torch.Tensor
    target_probs: torch.Tensor


def describe(name: str, probs: torch.Tensor) -> None:
    topv, topi = torch.topk(probs, k=min(5, probs.numel()))
    top = ", ".join(f"{int(i)}:{float(v):.4f}" for v, i in zip(topv, topi))
    print(f"{name:>20}: sum={float(probs.sum()):.6f} entropy={float(entropy(probs)):.6f} top=[{top}]")


def report_pair(pair: Pair) -> None:
    print(f"\n== {pair.name} ==")
    describe("input", pair.input_probs)
    describe("target", pair.target_probs)
    print(f"pesto CE          : {float(pesto_ce(pair.input_probs, pair.target_probs)):.6f}")
    print(f"pesto symmetric CE: {float(pesto_symmetric_ce(pair.input_probs, pair.target_probs)):.6f}")
    print(f"prob CE           : {float(prob_ce(pair.input_probs, pair.target_probs)):.6f}")
    print(f"prob symmetric CE : {float(prob_symmetric_ce(pair.input_probs, pair.target_probs)):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=219)
    parser.add_argument("--center", type=int, default=96)
    parser.add_argument("--shift", type=int, default=8)
    parser.add_argument("--pad-length", type=int, default=16)
    parser.add_argument("--sharpness", type=float, nargs="+", default=[3.0, 12.0, 40.0])
    args = parser.parse_args()

    bins = args.bins
    center = min(max(args.center, 0), bins - 1)
    shifted_center = min(max(center + args.shift, 0), bins - 1)

    print(f"bins={bins} log(bins)={math.log(bins):.6f} center={center} shift={args.shift}")
    print("Note: PESTO CE uses CrossEntropyLoss on already-softmaxed encoder outputs.")

    uniform = torch.full((bins,), 1.0 / bins)
    peak = one_hot(bins, center)
    shifted_peak = one_hot(bins, shifted_center)

    pairs = [
        Pair("uniform vs uniform", uniform, uniform),
        Pair("onehot match", peak, peak),
        Pair("onehot mismatch", peak, shifted_peak),
    ]

    for sharpness in args.sharpness:
        a = gaussian_probs(bins, center, sharpness)
        b = gaussian_probs(bins, center + 0.4, sharpness)
        c = gaussian_probs(bins, shifted_center, sharpness)
        pairs.extend([
            Pair(f"gauss s={sharpness:g} near", a, b),
            Pair(f"gauss s={sharpness:g} shifted", a, c),
        ])

    for pair in pairs:
        report_pair(pair)

    source = gaussian_probs(bins, center, args.sharpness[0])
    shifted = shift_distribution(source, args.shift)
    print("\n== ShiftCrossEntropy mechanics ==")
    describe("x1", source)
    describe("x2 shifted", shifted)
    print(f"pesto SCE aligned : {float(shift_cross_entropy_pesto(source, shifted, args.shift, args.pad_length)):.6f}")
    print(f"prob SCE aligned  : {float(shift_cross_entropy_prob(source, shifted, args.shift, args.pad_length)):.6f}")
    print(f"pesto SCE wrong   : {float(shift_cross_entropy_pesto(source, shifted, 0, args.pad_length)):.6f}")
    print(f"prob SCE wrong    : {float(shift_cross_entropy_prob(source, shifted, 0, args.pad_length)):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
