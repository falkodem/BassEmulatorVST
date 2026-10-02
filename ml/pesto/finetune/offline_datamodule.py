"""Offline HCQT frames with real past and future context for PESTO fine-tuning."""
import logging

import torch

from ml.pesto.finetune.streaming_datamodule import GuitarStreamingDataModule


log = logging.getLogger(__name__)


class GuitarOfflineDataModule(GuitarStreamingDataModule):
    frontend = "offline"

    def __init__(self, *, block_frames: int = 4096, **kwargs):
        super().__init__(**kwargs)
        if block_frames < 1:
            raise ValueError("block_frames must be positive")
        self.block_frames = block_frames

    @torch.no_grad()
    def _compute_hcqt_for_epoch(self, offset_samples: int) -> torch.Tensor:
        preproc = self._offline_preproc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preproc.to(device)

        context_samples = max(
            kernel.kernel_width for kernel in preproc.hcqt_kernels.cqt_kernels
        ) // 2
        audios = [audio[offset_samples:] for audio in self._audio_list]
        frame_counts = [audio.numel() // self.chunk_size for audio in audios]
        total_frames = sum(frame_counts)
        if not total_frames:
            raise ValueError("No complete HCQT frames in training WAVs")

        frames = torch.empty(
            total_frames,
            len(self.hcqt_kwargs["harmonics"]),
            self.hcqt_kwargs["n_bins"],
            2,
            dtype=torch.float32,
        )
        output_start = 0
        for file_idx, (audio, num_frames) in enumerate(zip(audios, frame_counts)):
            for first_frame in range(0, num_frames, self.block_frames):
                last_frame = min(num_frames, first_frame + self.block_frames)
                core_start = first_frame * self.chunk_size
                core_stop = last_frame * self.chunk_size
                segment_start = max(0, core_start - context_samples)
                segment_start -= segment_start % self.chunk_size
                segment_stop = min(audio.numel(), core_stop + context_samples)

                segment = audio[segment_start:segment_stop].unsqueeze(0).to(device)
                hcqt = preproc.hcqt(segment, sr=None).squeeze(0).permute(2, 0, 1, 3)
                local_start = first_frame - segment_start // self.chunk_size
                local_stop = local_start + last_frame - first_frame
                if local_stop > hcqt.size(0):
                    raise RuntimeError(
                        f"WAV {file_idx}: need HCQT frames through {local_stop}, "
                        f"got {hcqt.size(0)}"
                    )
                frames[output_start + first_frame:output_start + last_frame] = (
                    hcqt[local_start:local_stop].cpu()
                )
            output_start += num_frames
            log.info("WAV %d: %d offline HCQT frames", file_idx, num_frames)

        return frames
