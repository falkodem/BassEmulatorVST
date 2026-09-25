import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # Data
    data_version:  str
    target_sr:     int
    window_size:   int
    hop_size:      int

    # Split
    val_file_frac: float
    val_rnd_frac:  float
    seed:          int

    # Model
    architecture:  str

    # Training
    batch_size:    int
    epochs:        int
    lr:            float
    scheduler_patience: int
    scheduler_factor:   float

    # Output
    model_version: str

    # Loss — l1 | l2 | multi_scale_stft
    loss: str = "l1"

    # Domain / input transform — waveform | stft
    domain:          str = "waveform"
    stft_n_fft:      int = 512
    stft_hop_length: int = 128

    # Optional training extras (None = disabled)
    clip_grad_norm:          float | None = None
    early_stopping_patience: int   | None = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        with Path(path).open(encoding="utf-8") as f:
            return cls(**json.load(f))
