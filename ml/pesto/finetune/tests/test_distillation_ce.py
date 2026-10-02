import unittest

import torch
from torch import nn

from ml.pesto.finetune.config import DistillConfig
from ml.pesto.finetune.distillation_module import PESTODistillationModule


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hparams = {"output_dim": 5}
        self.weight = nn.Parameter(torch.zeros(1))


class DistillationCrossEntropyTests(unittest.TestCase):
    def test_upstream_mode_reaches_both_self_supervised_losses(self):
        module = PESTODistillationModule(
            encoder=_Encoder(),
            confidence=nn.Linear(5, 1),
            mode="pitch_kl_ssl",
            min_pitch_shift_steps=-2,
            max_pitch_shift_steps=2,
            bins_per_semitone=3,
            reduction="alwa",
            lr=1e-5,
            weight_decay=0.0,
            scheduler_epochs=1,
            ssl_cross_entropy="upstream",
            self_supervised_weights={
                "invariance": 0.2,
                "shift_entropy": 0.2,
                "equivariance": 0.1,
            },
        )

        self.assertEqual(module.invariance_loss.mode, "upstream")
        self.assertIs(module.shift_entropy_loss.criterion, module.invariance_loss)

    def test_default_remains_probability_mode(self):
        self.assertEqual(DistillConfig().ssl_cross_entropy, "probability")
        with self.assertRaises(ValueError):
            DistillConfig(ssl_cross_entropy="unknown")


if __name__ == "__main__":
    unittest.main()
