import unittest

import torch

from ml.pesto.finetune.offline_datamodule import GuitarOfflineDataModule


class OfflineFramesTests(unittest.TestCase):
    def test_blocks_match_whole_audio_hcqt(self):
        sample_rate = 44100
        hop = 441
        sample_count = 24 * hop + 137
        t = torch.arange(sample_count) / sample_rate
        audio = (0.5 * torch.sin(2 * torch.pi * 110 * t)).float()
        datamodule = GuitarOfflineDataModule(
            wav_paths=[], block_frames=5, random_offset=False,
            model_name="mir-1k_g7",
        )
        datamodule._audio_list = [audio]
        datamodule._offline_preproc = datamodule._build_offline_preprocessor()

        actual = datamodule._compute_hcqt_for_epoch(0)
        preproc = datamodule._offline_preproc
        device = next(preproc.parameters()).device
        expected = preproc.hcqt(audio.unsqueeze(0).to(device), sr=None)
        expected = expected.squeeze(0).permute(2, 0, 1, 3)[: sample_count // hop].cpu()

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
