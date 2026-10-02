import unittest

import torch
import torch.nn.functional as F

from ml.pesto.finetune.vendor.losses import CrossEntropyLoss, ShiftCrossEntropy


class ProbabilityCrossEntropyTests(unittest.TestCase):
    def test_perfect_one_hot_match_has_zero_loss(self):
        probabilities = torch.tensor([[0.0, 1.0, 0.0]])
        loss = CrossEntropyLoss()(probabilities, probabilities)
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_soft_targets_and_detached_target(self):
        prediction = torch.tensor([[0.8, 0.2]], requires_grad=True)
        target = torch.tensor([[0.25, 0.75]], requires_grad=True)
        loss = CrossEntropyLoss(detach_targets=True)(prediction, target)
        expected = -(target.detach() * prediction.detach().log()).sum()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertIsNone(target.grad)

    def test_symmetric_loss_averages_both_directions(self):
        first = torch.tensor([[0.8, 0.2]])
        second = torch.tensor([[0.25, 0.75]])
        loss = CrossEntropyLoss(symmetric=True, detach_targets=True)(first, second)
        expected = -0.5 * (
            (second * first.log()).sum() + (first * second.log()).sum()
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_aligned_shifted_one_hot_has_zero_loss(self):
        original = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        shifted = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        criterion = ShiftCrossEntropy(pad_length=2, criterion=CrossEntropyLoss())
        self.assertAlmostEqual(criterion(original, shifted, torch.tensor([1])).item(), 0.0)
        self.assertGreater(criterion(original, shifted, torch.tensor([0])).item(), 1.0)

    def test_upstream_mode_applies_second_softmax(self):
        prediction = torch.tensor([[0.8, 0.2]], requires_grad=True)
        target = torch.tensor([[0.25, 0.75]], requires_grad=True)
        loss = CrossEntropyLoss(mode="upstream", detach_targets=True)(prediction, target)
        self.assertTrue(torch.allclose(loss, F.cross_entropy(prediction, target.detach())))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertIsNone(target.grad)

    def test_shift_cross_entropy_uses_upstream_mode(self):
        original = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        shifted = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
        criterion = ShiftCrossEntropy(
            pad_length=2, criterion=CrossEntropyLoss(mode="upstream")
        )
        loss = criterion(original, shifted, torch.tensor([1]))
        aligned = F.pad(original, (2, 2))
        self.assertTrue(torch.allclose(loss, F.cross_entropy(aligned, aligned)))
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
