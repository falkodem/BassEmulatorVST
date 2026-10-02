import unittest

import torch

from ml.pesto.finetune.vendor.losses import ShiftWasserstein2


class ShiftWasserstein2Tests(unittest.TestCase):
    def setUp(self):
        self.loss = ShiftWasserstein2(pad_length=2)

    def test_known_shift_aligns_one_hot_distributions(self):
        original = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        shifted = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        self.assertAlmostEqual(self.loss(original, shifted, torch.tensor([1])).item(), 0.0)
        self.assertAlmostEqual(self.loss(original, shifted, torch.tensor([0])).item(), 1.0, delta=2e-4)

    def test_cost_scales_with_bin_distance(self):
        original = torch.tensor([[0.5, 0.0, 0.5, 0.0, 0.0]])
        shifted = torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]])
        self.assertAlmostEqual(self.loss(original, shifted, torch.tensor([0])).item(), 1.0, delta=2e-4)

    def test_shift_is_symmetric(self):
        original = torch.tensor([[0.2, 0.3, 0.4, 0.1, 0.0]])
        shifted = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]])
        forward = self.loss(original, shifted, torch.tensor([1]))
        reverse = self.loss(shifted, original, torch.tensor([-1]))
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-6))

    def test_gradient_reaches_both_distributions(self):
        logits1 = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        logits2 = torch.tensor([[-1.0, 1.0, 0.0]], requires_grad=True)
        loss = self.loss(logits1.softmax(-1), logits2.softmax(-1), torch.tensor([0]))
        loss.backward()
        for logits in (logits1, logits2):
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_matching_soft_distributions_have_finite_gradient(self):
        logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
        prediction = logits.softmax(-1)
        loss = self.loss(prediction, prediction, torch.tensor([0]))
        self.assertAlmostEqual(loss.item(), 0.0, delta=1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
