"""Fine-tune configuration for streaming PESTO.

Defaults match the plugin's runtime params:
 * sample_rate=44100, chunk_size=441, mirror=1.0, mirror_fn='refill'
 * HCQT params match the `mir-1k_g7` checkpoint (gamma=7, bps=3, n_bins=251)

For first experiment we use mirror=1.0+refill (zero added latency, idealistic
target). If quality is bad in plugin after fine-tune, drop to mirror=0.8.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def _default_wav_paths() -> list[str]:
    return [
        "/media/falkodem/VolumeD/Music/Projects/dataset/19-PESTO_0-260606_1408.wav",
        "/media/falkodem/VolumeD/Music/Projects/dataset/20-PESTO_1-260607_1530.wav",
        "/media/falkodem/VolumeD/Music/Projects/dataset/21-PESTO_2-260726_2105.wav",
        "/media/falkodem/VolumeD/Music/Projects/dataset/22-PESTO_3-260726_2117.wav",
    ]


@dataclass
class TrainConfig:
    # data
    wav_paths: Sequence[str] = field(default_factory=_default_wav_paths)
    sample_rate: int = 44100
    chunk_size: int = 441            # = step_size 10ms @ 44.1k
    random_offset: bool = True

    # HCQT (must match mir-1k_g7 — see pesto.weights.mir-1k_g7.ckpt hcqt_params)
    harmonics: Sequence[float] = (1,)
    fmin: float = 27.5
    bins_per_semitone: int = 3
    n_bins: int = 251
    center_bins: bool = True
    gamma: float = 7.

    # streaming (matches what the plugin sees)
    mirror: float = 1.0
    mirror_fn: str = "refill"

    # training
    batch_size: int = 512            # CQT frames per training step
    precompute_batch: int = 256      # parallel segments during HCQT precompute
    num_workers: int = 0             # CQT precompute is in main process
    lr: float = 1e-5                 # conservative for fine-tune from pretrained
    weight_decay: float = 0.0
    epochs: int = 80
    grad_clip: float = 3.0

    # loss weighting
    # "gradients" matches pesto-full: these weights are initial values, then
    # GradientsLossWeighting updates them from per-loss gradient norms every batch.
    # "fixed" keeps these weights constant for the whole run.
    loss_weighting: str = "gradients"   # "gradients" or "fixed"
    loss_weighting_ema: float = 0.999
    weight_invariance: float = 0.0
    weight_equivariance: float = 0.0
    weight_shift_entropy: float = 1.0

    # encoder arch (matches mir-1k_g7)
    # n_bins_in for encoder is what's left after PitchShiftCQT crop:
    #   pitch_shift_max_steps = bps * 11 // 2 = 16  (for bps=3)
    #   n_bins_in_encoder = n_bins - max_steps + min_steps = 251 - 16 - 16 = 219
    # Hmm but mir-1k_g7 was trained with n_bins=251 and PitchShiftCQT crops it
    # the same way → encoder n_bins_in = 251 - 32 = 219. Actually let's load
    # checkpoint hparams to confirm at runtime.
    # → these defaults are placeholder; train.py extracts from checkpoint.

    # checkpoint
    pretrained: str = "mir-1k_g7"    # name (in pesto.weights/) or path to .ckpt
    resume_from: str | None = None   # Lightning checkpoint to resume optimizer/scheduler/trainer state
    output_dir: str = "runs/finetune_pesto"
    run_name: str = ""               # auto-generated if empty (timestamp)

    # logging
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0  # validate every epoch

    # device
    accelerator: str = "auto"        # 'gpu' / 'cpu' / 'auto'
    devices: int = 1
    precision: str = "32-true"       # could try "16-mixed" for speed

    def __post_init__(self):
        self.wav_paths = [str(Path(p).expanduser()) for p in self.wav_paths]


@dataclass
class DistillConfig:
    """Configuration shared by teacher-label generation and distillation."""

    # data and fixed split
    wav_paths: Sequence[str] = field(default_factory=_default_wav_paths)
    validation_wav: str = (
        "/media/falkodem/VolumeD/Music/Projects/dataset/20-PESTO_1-260607_1530.wav"
    )
    validation_start_fraction: float = 0.5
    teacher_labels_dir: str = "data/pesto_teacher/mir-1k_g7"

    # teacher/student
    teacher_model: str = "mir-1k_g7"
    student_checkpoint: str = (
        "runs/finetune_pesto/20260706_110636/finetuned-20260706_110636.ckpt"
    )
    mode: str = "confidence"  # "confidence" or "pitch"

    # frame grid and streaming frontend
    sample_rate: int = 44100
    chunk_size: int = 441
    random_offset: bool = False
    mirror: float = 1.0
    mirror_fn: str = "refill"
    harmonics: Sequence[float] = (1,)
    fmin: float = 27.5
    bins_per_semitone: int = 3
    n_bins: int = 251
    center_bins: bool = True
    gamma: float = 7.0

    # teacher generation
    teacher_frames_per_block: int = 4096
    teacher_device: str = "auto"

    # training
    batch_size: int = 512
    precompute_batch: int = 256
    validation_precompute_batch: int = 64
    num_workers: int = 0
    confidence_lr: float = 1e-4
    pitch_lr: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 20
    grad_clip: float = 3.0
    teacher_confidence_power: float = 1.0

    # checkpoint/logging
    output_dir: str = "runs/distill_pesto"
    run_name: str = ""
    resume_from: str | None = None
    log_every_n_steps: int = 10

    # device
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"

    # smoke-test limiter; None uses complete ranges
    max_minutes: float | None = None

    def __post_init__(self):
        self.wav_paths = [str(Path(p).expanduser()) for p in self.wav_paths]
        self.validation_wav = str(Path(self.validation_wav).expanduser())
        self.teacher_labels_dir = str(Path(self.teacher_labels_dir).expanduser())
        self.student_checkpoint = str(Path(self.student_checkpoint).expanduser())
        if self.mode not in {"confidence", "pitch"}:
            raise ValueError(f"Unsupported distillation mode: {self.mode}")
        if not 0.0 < self.validation_start_fraction < 1.0:
            raise ValueError("validation_start_fraction must be between 0 and 1")
        if self.random_offset:
            raise ValueError("Distillation labels require random_offset=False")
