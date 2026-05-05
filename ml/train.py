#!/usr/bin/env python3
"""
Train a waveform/spectral model: guitar audio → bass audio.

Two validation sets:
  val_random  — random 10% windows from training files.
                Standard metric; used for checkpoint selection.
  val_by_file — one fully held-out file (unseen note).
                Generalization benchmark; not used for model selection.

Usage (Ubuntu):
  poetry install --with train
  poetry run python ml/train.py

TensorBoard:
  tensorboard --logdir runs/
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from losses import make_loss
from nn_architectures import REGISTRY
from train_config import TrainConfig
from transforms import make_transform

# ── Config ────────────────────────────────────────────────────────────────────

CFG = TrainConfig.load(Path(__file__).resolve().parent / "configs" / "train_v0.json")

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / CFG.data_version / "windows"

# Each launch gets its own dir: runs/{model_version}/{timestamp}/
# It holds: best.pt  config.json  events.out.tfevents.*
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / "runs" / CFG.model_version / RUN_TAG

# ── Dataset ───────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Yields (guitar, bass) tensor pairs from pre-sliced numpy arrays.

    If a transform is provided it is applied to both tensors, enabling
    domain switching (waveform → spectrogram) without changing the train loop.
    """

    def __init__(self, guitar: np.ndarray, bass: np.ndarray, transform=None):
        self.guitar    = torch.from_numpy(guitar[:, None, :])   # (N, 1, W)
        self.bass      = torch.from_numpy(bass[:, None, :])     # (N, 1, W)
        self.transform = transform

    def __len__(self):
        return len(self.guitar)

    def __getitem__(self, idx):
        g, b = self.guitar[idx], self.bass[idx]
        if self.transform is not None:
            g = self.transform(g)
            b = self.transform(b)
        return g, b


def make_splits(
    meta_csv: Path,
    val_file_frac: float,
    val_rnd_frac: float,
    seed: int,
) -> tuple[list, list, list, list]:
    """
    Returns (train_idx, val_random_idx, val_by_file_idx, held_files).

    val_by_file  — all windows from N held-out files (unseen notes).
    val_random   — random subset of remaining windows.
    train        — remaining windows after removing val_random.
    held_files   — sorted list of held-out source filenames.
    """
    with meta_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    files: dict[str, list[int]] = {}
    for row in rows:
        files.setdefault(row["source_file"], []).append(int(row["window_idx"]))

    file_list = sorted(files.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(file_list)

    n_held      = max(1, round(len(file_list) * val_file_frac))
    held_files  = sorted(file_list[:n_held])
    avail_files = file_list[n_held:]

    val_by_file_idx = [i for f in held_files  for i in files[f]]
    avail_idx       = [i for f in avail_files for i in files[f]]

    avail_arr = np.array(avail_idx)
    rng.shuffle(avail_arr)
    n_val_rnd      = max(1, round(len(avail_arr) * val_rnd_frac))
    val_random_idx = avail_arr[:n_val_rnd].tolist()
    train_idx      = avail_arr[n_val_rnd:].tolist()

    return train_idx, val_random_idx, val_by_file_idx, held_files

# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best     = float("inf")
        self.counter  = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# ── Training ──────────────────────────────────────────────────────────────────

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    clip_grad_norm: float | None = None,
) -> tuple[float, float]:
    """
    Returns (mean_loss, mean_grad_norm).

    grad_norm is the pre-clip L2 norm (useful for tuning clip_grad_norm).
    Returns 0.0 for validation.
    """
    model.train(train)
    total_loss  = 0.0
    total_gnorm = 0.0
    n_batches   = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for guitar, bass in loader:
            guitar, bass = guitar.to(device), bass.to(device)
            if train:
                optimizer.zero_grad()
            loss = criterion(model(guitar), bass)
            if train:
                loss.backward()
                gnorm = sum(
                    p.grad.detach().norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                total_gnorm += gnorm
                if clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            total_loss += loss.item() * len(guitar)
            n_batches  += 1

    mean_loss  = total_loss / len(loader.dataset)
    mean_gnorm = total_gnorm / n_batches if train else 0.0
    return mean_loss, mean_gnorm

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    print(f"\nRun dir: {RUN_DIR}")

    print("\nLoading dataset...")
    guitar = np.load(DATA_DIR / "guitar.npy")
    bass   = np.load(DATA_DIR / "bass.npy")
    print(f"  {guitar.shape[0]:,} windows  shape={guitar.shape}")

    print("\nSplitting...")
    train_idx, val_rnd_idx, val_file_idx, held_files = make_splits(
        DATA_DIR / "meta.csv", CFG.val_file_frac, CFG.val_rnd_frac, CFG.seed
    )

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  val_by_file held-out ({len(held_files)} file(s)):")
    for f in held_files:
        print(f"    • {f}")
    print(f"  Train:       {len(train_idx):,} windows")
    print(f"  val_random:  {len(val_rnd_idx):,} windows")
    print(f"  val_by_file: {len(val_file_idx):,} windows")
    print(sep)

    transform = make_transform(CFG.domain, CFG.stft_n_fft, CFG.stft_hop_length)
    full_ds   = WindowDataset(guitar, bass, transform=transform)
    pin       = device.type == "cuda"

    def make_loader(idx, shuffle):
        return DataLoader(
            Subset(full_ds, idx),
            batch_size=CFG.batch_size, shuffle=shuffle,
            num_workers=4, pin_memory=pin,
        )

    train_loader    = make_loader(train_idx,    shuffle=True)
    val_rnd_loader  = make_loader(val_rnd_idx,  shuffle=False)
    val_file_loader = make_loader(val_file_idx, shuffle=False)

    if CFG.architecture not in REGISTRY:
        raise ValueError(f"Unknown architecture '{CFG.architecture}'. Available: {list(REGISTRY)}")
    model     = REGISTRY[CFG.architecture]().to(device)
    criterion = make_loss(CFG.loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=CFG.scheduler_patience, factor=CFG.scheduler_factor
    )
    early_stopper = (
        EarlyStopping(CFG.early_stopping_patience)
        if CFG.early_stopping_patience is not None
        else None
    )

    CFG.save(RUN_DIR / "config.json")

    # One SummaryWriter → one tfevents file → one run in TensorBoard
    writer = SummaryWriter(log_dir=str(RUN_DIR))
    writer.add_text("config/loss",   CFG.loss,   0)
    writer.add_text("config/domain", CFG.domain, 0)
    writer.add_text(
        "val_by_file/held_out_files",
        "\n".join(f"- {f}" for f in held_files),
        global_step=0,
    )
    print(f"TensorBoard: tensorboard --logdir {ROOT / 'runs'}")

    extras = []
    if CFG.clip_grad_norm is not None:
        extras.append(f"clip={CFG.clip_grad_norm}")
    if CFG.early_stopping_patience is not None:
        extras.append(f"early_stop={CFG.early_stopping_patience}")
    extras_str = f"  [{', '.join(extras)}]" if extras else ""

    print(f"\nModel : {CFG.architecture}  params={model.param_count:,}")
    print(f"Loss  : {CFG.loss}  domain: {CFG.domain}{extras_str}")
    header = f"{'Epoch':>5}  {'Train':>10}  {'Val/rnd':>10}  {'Val/file':>10}  {'LR':>8}  {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    best_val = float("inf")

    for epoch in tqdm(range(1, CFG.epochs + 1), desc="Training", unit="epoch"):
        t0 = time.time()
        train_loss,    grad_norm  = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, clip_grad_norm=CFG.clip_grad_norm,
        )
        val_rnd_loss,  _          = run_epoch(model, val_rnd_loader,  criterion, optimizer, device, train=False)
        val_file_loss, _          = run_epoch(model, val_file_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_rnd_loss)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]

        # ── TensorBoard — add_scalar keeps one tfevents file per run ─────────
        writer.add_scalar("loss/train",    train_loss,    epoch)
        writer.add_scalar("loss/val_rnd",  val_rnd_loss,  epoch)
        writer.add_scalar("loss/val_file", val_file_loss, epoch)
        writer.add_scalar("train/grad_norm", grad_norm, epoch)
        writer.add_scalar("train/lr",        lr,        epoch)
        writer.add_scalar("time/epoch_sec",  elapsed,   epoch)

        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param.detach(), epoch)
        # ─────────────────────────────────────────────────────────────────────

        tqdm.write(
            f"{epoch:5d}  {train_loss:10.6f}  {val_rnd_loss:10.6f}"
            f"  {val_file_loss:10.6f}  {lr:8.2e}  {elapsed:5.1f}s"
        )

        if val_rnd_loss < best_val:
            best_val = val_rnd_loss
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "val_random_loss": val_rnd_loss,
                    "val_file_loss":   val_file_loss,
                },
                RUN_DIR / "best.pt",
            )
            tqdm.write(f"        ^ saved best  (val_rnd={val_rnd_loss:.6f}  val_file={val_file_loss:.6f})")

        if early_stopper is not None and early_stopper.step(val_rnd_loss):
            tqdm.write(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {CFG.early_stopping_patience} epochs)"
            )
            break

    writer.close()
    print(f"\nDone. Best val_random: {best_val:.6f}")
    print(f"Checkpoint : {RUN_DIR / 'best.pt'}")
    print(f"TensorBoard: tensorboard --logdir {ROOT / 'runs'}")


if __name__ == "__main__":
    main()
