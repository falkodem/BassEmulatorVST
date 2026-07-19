"""Streaming HCQT DataModule for fine-tuning PESTO under realtime conditions.

Key idea: produce CQT frames that look EXACTLY like what the plugin sees at
inference time. We do this by running each chunk of audio through `CachedConv1d`
(left pad = cache of real previous samples, right pad = `mirror_fn` fake), same
code path as the ONNX export.

Why not just use pesto-full's AudioDataModule:
 * pesto-full computes CQT in offline mode (whole-file reflect-pad) → training
   distribution doesn't match streaming inference
 * pesto-full uses nnAudio CQT without `gamma` parameter → can't match our
   `mir-1k_g7` checkpoint which was trained with `gamma=7`

Implementation:
 1. Load all WAV files once into a list of CPU tensors
 2. Each epoch: pick a random sample offset in [0, hop_length), then build CQT
    for the whole dataset by chunking and running streaming CQT batched across
    `precompute_batch` parallel segments. Result is a (N_frames, H, F, 2) tensor
    cached in RAM (a few GB) for the duration of one epoch
 3. DataLoader iterates over frames; per-frame transforms (ToLogMagnitude etc.)
    happen in `on_after_batch_transfer`
 4. PyTorch Lightning re-calls `train_dataloader()` each epoch when
    `reload_dataloaders_every_n_epochs=1`, which triggers a new offset + new CQT

Border artifacts: when we batch by N parallel segments, the first ~9 chunks of
each segment have cache=zeros instead of real prior audio. These ~0.5% "warmup"
frames are kept in training — they're exactly what the plugin sees on startup,
so the model should learn to handle them.
"""
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data
import pytorch_lightning as pl

from pesto.loader import load_model
from pesto.utils.cached_conv import CachedConv1d, RefillPad1d


log = logging.getLogger(__name__)


class _ComplexFramesDataset(torch.utils.data.Dataset):
    """Yields complex CQT frames from a (N, H, F, 2) float tensor."""
    def __init__(self, inputs: torch.Tensor):
        # inputs shape: (N_frames, harmonics, freqs, 2)  float32
        assert inputs.ndim == 4 and inputs.size(-1) == 2, \
            f"expected (N, H, F, 2) real/imag, got {tuple(inputs.shape)}"
        self.inputs = inputs

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, item):
        # view_as_complex requires contiguous + last dim == 2
        frame = self.inputs[item]
        return torch.view_as_complex(frame.contiguous()), 0  # dummy label


class GuitarStreamingDataModule(pl.LightningDataModule):
    """Self-supervised PESTO training data with realtime-streaming HCQT.

    Args:
        wav_paths: list of audio file paths (mono, sample_rate Hz)
        sample_rate: must match the rate the plugin runs at (44100)
        chunk_size: samples per inference call (441 = 10 ms hop)
        precompute_batch: how many parallel segments to batch when computing HCQT
        batch_size: DataLoader batch size (CQT frames per training step)
        random_offset: shift the global chunk grid by a random offset each epoch
        # HCQT params (must match the pretrained checkpoint exactly)
        harmonics, fmin, fmax, bins_per_semitone, n_bins, center_bins, gamma
        # streaming
        mirror, mirror_fn ('zeros' or 'refill')
        # model checkpoint to take HCQT kernels from
        model_name: pesto checkpoint name (e.g. 'mir-1k_g7')
    """
    def __init__(self,
                 wav_paths: Sequence[str | Path],
                 *,
                 sample_rate: int = 44100,
                 chunk_size: int = 441,
                 precompute_batch: int = 256,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 random_offset: bool = True,
                 # HCQT (must match mir-1k_g7 hcqt_params)
                 harmonics: Sequence[float] = (1,),
                 fmin: float = 27.5,
                 fmax: float | None = None,
                 bins_per_semitone: int = 3,
                 n_bins: int = 251,
                 center_bins: bool = True,
                 gamma: float = 7.,
                 # streaming
                 mirror: float = 1.0,
                 mirror_fn: str = 'refill',
                 # source for kernels
                 model_name: str = 'mir-1k_g7',
                 transforms: Sequence[nn.Module] | None = None):
        super().__init__()
        self.wav_paths = [Path(p) for p in wav_paths]
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.precompute_batch = precompute_batch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_offset = random_offset
        self.model_name = model_name

        self.hcqt_kwargs = dict(
            harmonics=list(harmonics),
            fmin=fmin,
            fmax=fmax,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins,
            gamma=gamma,
        )
        self.hop_duration = 1000.0 * chunk_size / sample_rate  # ms — for checkpoint compat

        self.mirror = mirror
        self.mirror_fn = mirror_fn

        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()

        # state filled in setup()
        self._audio_list: list[torch.Tensor] | None = None
        # streaming preprocessor (created in setup, lives on data device)
        self._stream_preproc: nn.Module | None = None
        # offline preprocessor for estimate_shift (small synth inputs)
        self._offline_preproc: nn.Module | None = None
        # epoch counter for logging
        self._epoch = 0

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        if self._audio_list is not None:
            return  # already done

        log.info("Loading %d WAV files into RAM", len(self.wav_paths))
        audios = []
        total_samples = 0
        for path in self.wav_paths:
            data, sr = sf.read(str(path), dtype='float32', always_2d=False)
            if sr != self.sample_rate:
                raise ValueError(f"{path.name}: sample_rate={sr}, expected {self.sample_rate}")
            if data.ndim > 1:
                data = data.mean(axis=1)
            audios.append(torch.from_numpy(data))
            total_samples += len(data)
        self._audio_list = audios
        total_min = total_samples / self.sample_rate / 60
        log.info("Loaded %.2f minutes total (%d samples)", total_min, total_samples)

        # build streaming preprocessor (will be moved to GPU lazily)
        self._stream_preproc = self._build_streaming_preprocessor()
        self._offline_preproc = self._build_offline_preprocessor()

    def _build_streaming_preprocessor(self) -> nn.Module:
        m = load_model(
            self.model_name,
            step_size=self.hop_duration,
            sampling_rate=self.sample_rate,
            streaming=True,
            max_batch_size=self.precompute_batch,
            mirror=self.mirror,
        )
        preproc = m.preprocessor
        if self.mirror_fn == 'refill':
            for _, mod in preproc.named_modules():
                if isinstance(mod, CachedConv1d):
                    right = mod.mirror.padding[1]
                    if right > 0:
                        mod.mirror = RefillPad1d((0, right))
        return preproc

    def _build_offline_preprocessor(self) -> nn.Module:
        """For estimate_shift only — uses offline (reflect-pad) CQT on small synth audio."""
        m = load_model(
            self.model_name,
            step_size=self.hop_duration,
            sampling_rate=self.sample_rate,
        )
        return m.preprocessor

    # ── per-epoch HCQT precompute ──────────────────────────────────────────

    @torch.no_grad()
    def _compute_hcqt_for_epoch(self, offset_samples: int) -> torch.Tensor:
        """Run streaming CQT over all audio with `offset_samples` head-skip.

        Returns CQT tensor shape (N_frames_total, harmonics, freqs, 2) on CPU.
        """
        # Move preprocessor to the device where data will be computed.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._stream_preproc.to(device)

        all_frames = []
        for file_idx, audio in enumerate(self._audio_list):
            # apply offset, then trim to multiple of (precompute_batch * chunk_size)
            audio = audio[offset_samples:]
            n_chunks_total = audio.numel() // self.chunk_size
            seg_len_chunks = n_chunks_total // self.precompute_batch
            if seg_len_chunks < 1:
                log.warning("File %d too short for batch=%d, skipping",
                            file_idx, self.precompute_batch)
                continue
            usable_samples = self.precompute_batch * seg_len_chunks * self.chunk_size
            audio = audio[:usable_samples].to(device)
            audio_b = audio.view(self.precompute_batch, seg_len_chunks * self.chunk_size)

            # reset cache to zeros (per-file fresh start — segments start with zero cache)
            self._reset_streaming_cache(device)

            # sequential by chunk index within each segment, batched across segments
            for ci in range(seg_len_chunks):
                chunk = audio_b[:, ci * self.chunk_size: (ci + 1) * self.chunk_size]
                # preprocessor expects (batch, samples). Output: depends — preprocessor returns log-magnitude,
                # but we want the raw complex HCQT so transforms ToLogMagnitude can run later.
                # → directly call hcqt() to get the complex (B, H, F, T, 2) tensor.
                hcqt = self._stream_preproc.hcqt(chunk, sr=None)  # (B, H, F, T=1, 2)
                # shape: (precompute_batch, n_harmonics, n_bins, 1, 2)
                # → flatten time, want one frame per call: take T=0
                frame = hcqt.squeeze(3)  # (B, H, F, 2)
                all_frames.append(frame.cpu())

        # Stack: each element is (B, H, F, 2). We want (N_total_frames, H, F, 2).
        # Order doesn't matter (DataLoader shuffles), but inside one segment frames are sequential.
        out = torch.cat([f for f in all_frames], dim=0)
        # Result is (sum of B across appends, H, F, 2) — each "B" entry is one frame
        # of one of the precompute_batch segments at one chunk index.
        return out

    def _reset_streaming_cache(self, device: str) -> None:
        for mod in self._stream_preproc.modules():
            if isinstance(mod, CachedConv1d) and hasattr(mod.cache, 'pad'):
                mod.cache.pad = torch.zeros_like(mod.cache.pad, device=device)

    # ── DataLoaders ────────────────────────────────────────────────────────

    def train_dataloader(self):
        """Called every epoch when `reload_dataloaders_every_n_epochs=1`."""
        offset = random.randint(0, self.chunk_size - 1) if self.random_offset else 0
        log.info("Epoch %d: building HCQT with offset=%d samples", self._epoch, offset)
        cqt = self._compute_hcqt_for_epoch(offset)
        log.info("  HCQT shape=%s, %.1f MB",
                 tuple(cqt.shape), cqt.numel() * 4 / 1e6)
        self._epoch += 1

        dataset = _ComplexFramesDataset(cqt)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        """Dummy validation set — PESTO requires one for estimate_shift to fire."""
        # one batch of zeros, just to trigger on_validation_epoch_start (estimate_shift)
        dummy = torch.zeros(1, len(self.hcqt_kwargs['harmonics']),
                             self.hcqt_kwargs['n_bins'], 2)
        return torch.utils.data.DataLoader(_ComplexFramesDataset(dummy), batch_size=1)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        x, y = batch
        return self.transforms(x), y

    # ── interface expected by PESTO.estimate_shift ────────────────────────

    def hcqt(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Offline HCQT for short synth audio (used in PESTO.estimate_shift).

        Mirrors AudioDataModule.hcqt() signature/output:
        returns (time, harmonics, freqs, 2) — non-streaming, reflect-padded.
        """
        # The offline preprocessor was built for sample_rate=self.sample_rate.
        # estimate_shift calls with sr=16000 — we need to rebuild for that sr.
        # Use a temp preprocessor with the right sr.
        if sr != self.sample_rate:
            m = load_model(self.model_name, step_size=self.hop_duration, sampling_rate=sr)
            preproc = m.preprocessor
        else:
            preproc = self._offline_preproc

        device = next(preproc.parameters(), torch.tensor(0)).device
        audio = audio.to(device)
        # hcqt method returns (B, H, F, T, 2). We want (T, H, F, 2) as in upstream.
        complex_cqt = preproc.hcqt(audio.unsqueeze(0), sr=None)  # (1, H, F, T, 2)
        return complex_cqt.squeeze(0).permute(2, 0, 1, 3)  # (T, H, F, 2)
