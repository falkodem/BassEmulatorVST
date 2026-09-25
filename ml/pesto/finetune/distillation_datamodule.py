"""Streaming HCQT data paired with frame-aligned offline PESTO targets."""
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
import torch.utils.data
from pesto.loader import load_model
from pesto.utils.cached_conv import CachedConv1d, RefillPad1d


log = logging.getLogger(__name__)


class ToLogMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 2:
            x = torch.view_as_complex(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return x.abs().clamp_min(self.eps).log10().mul(20.0)


@dataclass(frozen=True)
class FrameRange:
    wav_path: Path
    labels_path: Path
    first_frame: int
    last_frame: int

    @property
    def num_frames(self) -> int:
        return self.last_frame - self.first_frame


class DistillationFramesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cqt: torch.Tensor,
        teacher_activations: torch.Tensor,
        teacher_confidence: torch.Tensor,
        valid: torch.Tensor,
    ):
        size = cqt.size(0)
        if not (
            teacher_activations.size(0) == size
            and teacher_confidence.size(0) == size
            and valid.size(0) == size
        ):
            raise ValueError("CQT and teacher target lengths do not match")
        self.cqt = cqt
        self.teacher_activations = teacher_activations
        self.teacher_confidence = teacher_confidence
        self.valid = valid

    def __len__(self) -> int:
        return self.cqt.size(0)

    def __getitem__(self, index: int):
        frame = torch.view_as_complex(self.cqt[index].contiguous())
        return (
            frame,
            self.teacher_activations[index],
            self.teacher_confidence[index],
            self.valid[index],
        )


class PESTODistillationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wav_paths: Sequence[str | Path],
        validation_wav: str | Path,
        teacher_labels_dir: str | Path,
        *,
        validation_start_fraction: float = 0.5,
        sample_rate: int = 44100,
        chunk_size: int = 441,
        precompute_batch: int = 256,
        validation_precompute_batch: int = 64,
        batch_size: int = 512,
        num_workers: int = 0,
        mirror: float = 1.0,
        mirror_fn: str = "refill",
        model_name: str = "mir-1k_g7",
        max_minutes: float | None = None,
    ):
        super().__init__()
        self.wav_paths = [Path(path).resolve() for path in wav_paths]
        self.validation_wav = Path(validation_wav).resolve()
        self.teacher_labels_dir = Path(teacher_labels_dir)
        self.validation_start_fraction = validation_start_fraction
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.precompute_batch = precompute_batch
        self.validation_precompute_batch = validation_precompute_batch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mirror = mirror
        self.mirror_fn = mirror_fn
        self.model_name = model_name
        self.max_minutes = max_minutes
        self.hop_duration = 1000.0 * chunk_size / sample_rate
        self.transforms = ToLogMagnitude()

        self.hcqt_kwargs: dict | None = None
        self._stream_preproc: nn.Module | None = None
        self._train_ranges: list[FrameRange] = []
        self._val_ranges: list[FrameRange] = []
        self._train_dataset: DistillationFramesDataset | None = None
        self._val_dataset: DistillationFramesDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._stream_preproc is not None:
            return
        manifest_path = self.teacher_labels_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Teacher manifest not found: {manifest_path}. "
                "Run generate_teacher_labels first."
            )
        manifest = json.loads(manifest_path.read_text())
        if int(manifest["sample_rate"]) != self.sample_rate:
            raise ValueError("Teacher-label sample rate does not match distillation config")
        if int(manifest["hop_samples"]) != self.chunk_size:
            raise ValueError("Teacher-label hop does not match distillation config")

        entries = {
            Path(item["source_path"]).resolve(): item
            for item in manifest["files"]
        }
        max_frames = None
        if self.max_minutes is not None:
            max_frames = int(self.max_minutes * 60.0 * self.sample_rate) // self.chunk_size

        validation_found = False
        for wav_path in self.wav_paths:
            if not wav_path.is_file():
                raise FileNotFoundError(f"Training WAV not found: {wav_path}")
            if wav_path not in entries:
                raise ValueError(f"No teacher labels registered for {wav_path}")
            item = entries[wav_path]
            labels_path = self.teacher_labels_dir / item["labels_file"]
            available = int(item["num_frames"])
            source_frames = sf.info(wav_path).frames // self.chunk_size
            if available < source_frames:
                raise ValueError(
                    f"{wav_path.name}: labels contain {available} frames, "
                    f"but full distillation needs {source_frames}"
                )

            if wav_path == self.validation_wav:
                validation_found = True
                split = int(source_frames * self.validation_start_fraction)
                train_stop = split if max_frames is None else min(split, max_frames)
                val_stop = source_frames if max_frames is None else min(
                    source_frames, split + max_frames
                )
                self._train_ranges.append(FrameRange(wav_path, labels_path, 0, train_stop))
                self._val_ranges.append(FrameRange(wav_path, labels_path, split, val_stop))
            else:
                stop = source_frames if max_frames is None else min(source_frames, max_frames)
                self._train_ranges.append(FrameRange(wav_path, labels_path, 0, stop))

        if not validation_found:
            raise ValueError(f"validation_wav is not present in wav_paths: {self.validation_wav}")

        model = load_model(
            self.model_name,
            step_size=self.hop_duration,
            sampling_rate=self.sample_rate,
            streaming=True,
            max_batch_size=self.precompute_batch,
            mirror=self.mirror,
        )
        self._stream_preproc = model.preprocessor
        if self.mirror_fn == "refill":
            for module in self._stream_preproc.modules():
                if isinstance(module, CachedConv1d):
                    right = module.mirror.padding[1]
                    if right:
                        module.mirror = RefillPad1d((0, right))
        elif self.mirror_fn != "zeros":
            raise ValueError(f"Unsupported mirror_fn: {self.mirror_fn}")
        runtime_keys = {"streaming", "mirror", "max_batch_size"}
        self.hcqt_kwargs = {
            key: value
            for key, value in model.preprocessor.hcqt_kwargs.items()
            if key not in runtime_keys
        }

        log.info(
            "Distillation split: train=%d frames, validation=%d frames",
            sum(item.num_frames for item in self._train_ranges),
            sum(item.num_frames for item in self._val_ranges),
        )

    def _reset_cache(self, device: torch.device) -> None:
        assert self._stream_preproc is not None
        for module in self._stream_preproc.modules():
            if isinstance(module, CachedConv1d) and hasattr(module.cache, "pad"):
                module.cache.pad = torch.zeros_like(module.cache.pad, device=device)

    def _carry_last_cache_to_first(self, batch_size: int) -> None:
        if batch_size <= 1:
            return
        assert self._stream_preproc is not None
        for module in self._stream_preproc.modules():
            if isinstance(module, CachedConv1d) and hasattr(module.cache, "pad"):
                module.cache.pad[0].copy_(module.cache.pad[batch_size - 1])

    def _warmup_chunks(self) -> int:
        assert self._stream_preproc is not None
        padding = max(
            (
                module.cache.padding
                for module in self._stream_preproc.modules()
                if isinstance(module, CachedConv1d)
            ),
            default=0,
        )
        return math.ceil(padding / self.chunk_size)

    @staticmethod
    def _load_targets(frame_range: FrameRange) -> tuple[np.ndarray, np.ndarray]:
        with np.load(frame_range.labels_path) as labels:
            start, stop = frame_range.first_frame, frame_range.last_frame
            activations = labels["activations"][start:stop].copy()
            confidence = labels["confidence"][start:stop].copy()
        return activations, confidence

    @staticmethod
    def _read_range(
        frame_range: FrameRange,
        chunk_size: int,
        history_frames: int = 0,
    ) -> tuple[np.ndarray, int]:
        history_start = max(0, frame_range.first_frame - history_frames)
        start_sample = history_start * chunk_size
        stop_sample = frame_range.last_frame * chunk_size
        with sf.SoundFile(frame_range.wav_path) as stream:
            stream.seek(start_sample)
            audio = stream.read(
                stop_sample - start_sample, dtype="float32", always_2d=True
            )
        if audio.shape[0] != stop_sample - start_sample:
            raise RuntimeError(f"Short read from {frame_range.wav_path}")
        return audio.mean(axis=1, dtype=np.float32), frame_range.first_frame - history_start

    @torch.no_grad()
    def _compute_train_range(
        self,
        frame_range: FrameRange,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        assert self._stream_preproc is not None
        teacher_acts, teacher_conf = self._load_targets(frame_range)
        audio_np, _ = self._read_range(frame_range, self.chunk_size)
        audio = torch.from_numpy(audio_np).to(device)
        num_frames = frame_range.num_frames
        parallel = min(self.precompute_batch, num_frames)
        segment_frames = num_frames // parallel
        main_frames = parallel * segment_frames
        warmup = self._warmup_chunks()

        frames_out: list[torch.Tensor] = []
        acts_out: list[torch.Tensor] = []
        conf_out: list[torch.Tensor] = []
        valid_out: list[torch.Tensor] = []

        self._reset_cache(device)
        main_audio = audio[:main_frames * self.chunk_size].view(
            parallel, segment_frames * self.chunk_size
        )
        segment_offsets = torch.arange(parallel) * segment_frames
        for local_frame in range(segment_frames):
            chunk = main_audio[
                :, local_frame * self.chunk_size:(local_frame + 1) * self.chunk_size
            ]
            hcqt = self._stream_preproc.hcqt(chunk, sr=None).squeeze(3)
            indices = (segment_offsets + local_frame).cpu().numpy()
            valid = torch.ones(parallel, dtype=torch.bool)
            if local_frame < warmup and parallel > 1:
                valid[1:] = False
            frames_out.append(hcqt.cpu())
            acts_out.append(torch.from_numpy(teacher_acts[indices]))
            conf_out.append(torch.from_numpy(teacher_conf[indices]))
            valid_out.append(valid)

        if main_frames < num_frames:
            self._carry_last_cache_to_first(parallel)
            for index in range(main_frames, num_frames):
                chunk = audio[index * self.chunk_size:(index + 1) * self.chunk_size].unsqueeze(0)
                hcqt = self._stream_preproc.hcqt(chunk, sr=None).squeeze(3)
                frames_out.append(hcqt.cpu())
                acts_out.append(torch.from_numpy(teacher_acts[index:index + 1]))
                conf_out.append(torch.from_numpy(teacher_conf[index:index + 1]))
                valid_out.append(torch.ones(1, dtype=torch.bool))

        return frames_out, acts_out, conf_out, valid_out

    @torch.no_grad()
    def _compute_validation_range(
        self,
        frame_range: FrameRange,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        assert self._stream_preproc is not None
        teacher_acts, teacher_conf = self._load_targets(frame_range)
        num_frames = frame_range.num_frames
        parallel = min(self.validation_precompute_batch, num_frames)
        base_length, extra = divmod(num_frames, parallel)
        lengths = np.asarray(
            [base_length + (index < extra) for index in range(parallel)],
            dtype=np.int64,
        )
        offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
        history = self._warmup_chunks()
        row_frames = history + int(lengths.max())
        audio = np.zeros(
            (parallel, row_frames * self.chunk_size), dtype=np.float32
        )

        with sf.SoundFile(frame_range.wav_path) as stream:
            for row, (offset, length) in enumerate(zip(offsets, lengths)):
                target_start = frame_range.first_frame + int(offset)
                available_history = min(history, target_start)
                read_start = target_start - available_history
                read_frames = available_history + int(length)
                stream.seek(read_start * self.chunk_size)
                samples = stream.read(
                    read_frames * self.chunk_size,
                    dtype="float32",
                    always_2d=True,
                ).mean(axis=1, dtype=np.float32)
                destination = (history - available_history) * self.chunk_size
                audio[row, destination:destination + samples.size] = samples

        audio_tensor = torch.from_numpy(audio).to(device)
        self._reset_cache(device)
        frames_out: list[torch.Tensor] = []
        acts_out: list[torch.Tensor] = []
        conf_out: list[torch.Tensor] = []
        valid_out: list[torch.Tensor] = []
        for step in range(row_frames):
            start = step * self.chunk_size
            chunk = audio_tensor[:, start:start + self.chunk_size]
            hcqt = self._stream_preproc.hcqt(chunk, sr=None).squeeze(3)
            local_frame = step - history
            if local_frame < 0:
                continue
            active = np.flatnonzero(local_frame < lengths)
            indices = offsets[active] + local_frame
            frames_out.append(hcqt[active].cpu())
            acts_out.append(torch.from_numpy(teacher_acts[indices]))
            conf_out.append(torch.from_numpy(teacher_conf[indices]))
            valid_out.append(torch.ones(active.size, dtype=torch.bool))

        return frames_out, acts_out, conf_out, valid_out

    def _build_dataset(
        self,
        ranges: Sequence[FrameRange],
        *,
        validation: bool,
    ) -> DistillationFramesDataset:
        assert self._stream_preproc is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._stream_preproc.to(device)
        frame_parts: list[torch.Tensor] = []
        acts_parts: list[torch.Tensor] = []
        conf_parts: list[torch.Tensor] = []
        valid_parts: list[torch.Tensor] = []

        for frame_range in ranges:
            log.info(
                "Precomputing %s frames %d:%d",
                frame_range.wav_path.name,
                frame_range.first_frame,
                frame_range.last_frame,
            )
            parts = (
                self._compute_validation_range(frame_range, device)
                if validation
                else self._compute_train_range(frame_range, device)
            )
            frame_parts.extend(parts[0])
            acts_parts.extend(parts[1])
            conf_parts.extend(parts[2])
            valid_parts.extend(parts[3])

        cqt = torch.cat(frame_parts)
        acts = torch.cat(acts_parts)
        confidence = torch.cat(conf_parts)
        valid = torch.cat(valid_parts)
        log.info(
            "Prepared %s: CQT=%s (%.1f MB), valid=%d/%d",
            "validation" if validation else "train",
            tuple(cqt.shape),
            cqt.numel() * cqt.element_size() / 1e6,
            int(valid.sum()),
            valid.numel(),
        )
        return DistillationFramesDataset(cqt, acts, confidence, valid)

    def train_dataloader(self):
        if self._train_dataset is None:
            self._train_dataset = self._build_dataset(self._train_ranges, validation=False)
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self):
        if self._val_dataset is None:
            self._val_dataset = self._build_dataset(self._val_ranges, validation=True)
        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        x, teacher_acts, teacher_conf, valid = batch
        return self.transforms(x), teacher_acts, teacher_conf, valid

