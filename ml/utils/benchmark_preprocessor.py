"""Бенчмарк скорости HCQT-препроцессинга для fine-tune'а PESTO.

Тестирует три варианта подсчёта CQT-фреймов на одинаковом куске аудио:

  (A) STREAMING sequential, batch=1 — наивная симуляция плагина: каждый
      chunk=441 семплов отдельным forward'ом, cache сшивается между ними.
      Идеально воспроизводит то, что увидит ONNX в плагине, но медленно
      из-за overhead'а 559k forward-launch'ей.

  (B) OFFLINE whole-file — как делает pesto-full сейчас: один forward на
      весь файл, reflect-pad по краям. Быстро, но это offline-распределение,
      на котором обучать realtime-модель неправильно.

  (C) STREAMING batched по N параллельных сегментов файла — компромисс:
      режем файл на N равных кусков, batch'им их через CachedConv1d с
      max_batch_size=N. Каждая позиция в батче имеет свой cache, сшивается
      внутри сегмента. На границах сегментов cache не передаётся (warmup
      из нулей ~88 мс), но это мизерная погрешность относительно всего
      файла: N стыков из ~50000 chunks.

Цель замеров — понять, реалистично ли on-the-fly CQT в DataLoader (с random
chunk-offset каждую эпоху для аугментации), или нужно precompute'ить и
кэшировать.
"""
import argparse
import time

import numpy as np
import soundfile as sf
import torch

from pesto.loader import load_model
from pesto.utils.cached_conv import CachedConv1d, RefillPad1d


SR = 44100
CHUNK = 441


def make_streaming_preproc(device: str,
                           max_batch_size: int,
                           mirror: float,
                           mirror_fn: str) -> torch.nn.Module:
    """Streaming-preprocessor (CachedConv1d) на нужном устройстве с нулевым cache."""
    m = load_model('mir-1k_g7', step_size=10.0, sampling_rate=SR,
                   streaming=True, max_batch_size=max_batch_size, mirror=mirror)
    preproc = m.preprocessor
    if mirror_fn == 'refill':
        for _, mod in preproc.named_modules():
            if isinstance(mod, CachedConv1d):
                right = mod.mirror.padding[1]
                mod.mirror = RefillPad1d((0, right))
    preproc = preproc.to(device)
    for mod in preproc.modules():
        if isinstance(mod, CachedConv1d) and hasattr(mod.cache, 'pad'):
            mod.cache.pad = torch.zeros_like(mod.cache.pad, device=device)
    return preproc


def reset_cache(preproc: torch.nn.Module, device: str) -> None:
    for mod in preproc.modules():
        if isinstance(mod, CachedConv1d):
            mod.cache.pad = torch.zeros_like(mod.cache.pad, device=device)


def sync(device: str) -> None:
    if device == 'cuda':
        torch.cuda.synchronize()


def bench_sequential(audio_t: torch.Tensor, n_chunks: int, device: str) -> None:
    print('=== (A) STREAMING sequential, chunk=441, batch=1 ===')
    preproc = make_streaming_preproc(device, max_batch_size=1, mirror=1.0, mirror_fn='refill')

    with torch.no_grad():
        for i in range(20):
            c = audio_t[i*CHUNK:(i+1)*CHUNK].unsqueeze(0)
            _ = preproc(c, sr=None)
    sync(device)
    reset_cache(preproc, device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(n_chunks):
            c = audio_t[i*CHUNK:(i+1)*CHUNK].unsqueeze(0)
            _ = preproc(c, sr=None)
    sync(device)
    dt = time.time() - t0
    print(f'  total {dt:.2f}s ({1000*dt/n_chunks:.3f} ms/chunk, {n_chunks/dt:.0f} chunks/s)')


def bench_offline(audio_t: torch.Tensor, device: str) -> None:
    print('=== (B) OFFLINE whole-file CQT (без streaming) ===')
    m_off = load_model('mir-1k_g7', step_size=10.0, sampling_rate=SR)
    preproc_off = m_off.preprocessor.to(device)
    audio_off = audio_t.unsqueeze(0)

    with torch.no_grad():
        _ = preproc_off(audio_off[:, :SR], sr=None)
    sync(device)

    t0 = time.time()
    with torch.no_grad():
        hcqt_off = preproc_off(audio_off, sr=None)
    sync(device)
    dt = time.time() - t0
    print(f'  total {dt:.3f}s, output shape={tuple(hcqt_off.shape)}')


MIN_SEG_CHUNKS = 8192 // CHUNK + 1  # need at least kernel_width samples per segment

def bench_batched(audio_t: torch.Tensor, n_chunks: int, device: str,
                  batch_sizes=(16, 64, 256)) -> None:
    print('=== (C) STREAMING batched по N параллельных сегментов файла ===')
    for batch in batch_sizes:
        seg_len_chunks = n_chunks // batch
        if seg_len_chunks < MIN_SEG_CHUNKS:
            print(f'  BATCH={batch:3d}: skip — segment too short '
                  f'({seg_len_chunks} chunks, need >={MIN_SEG_CHUNKS})')
            continue
        seg_len_samp = seg_len_chunks * CHUNK
        audio_batched = audio_t[:batch*seg_len_samp].view(batch, seg_len_samp)

        preproc_b = make_streaming_preproc(device, max_batch_size=batch,
                                           mirror=1.0, mirror_fn='refill')

        with torch.no_grad():
            for i in range(5):
                c = audio_batched[:, i*CHUNK:(i+1)*CHUNK]
                _ = preproc_b(c, sr=None)
        sync(device)
        reset_cache(preproc_b, device)

        t0 = time.time()
        with torch.no_grad():
            for i in range(seg_len_chunks):
                c = audio_batched[:, i*CHUNK:(i+1)*CHUNK]
                _ = preproc_b(c, sr=None)
        sync(device)
        dt = time.time() - t0
        total_frames = batch * seg_len_chunks
        print(f'  BATCH={batch:3d}: {dt:.2f}s '
              f'({1000*dt/total_frames:.3f} ms/frame, {total_frames/dt:.0f} frames/s)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', default='/media/falkodem/VolumeD/Music/Projects/dataset/19-PESTO_0-260606_1408.wav')
    parser.add_argument('--seconds', type=float, default=8 * 60,
                        help='Length of audio segment to benchmark (default 8 min)')
    parser.add_argument('--skip-sequential', action='store_true',
                        help='Skip the slow batch=1 sequential variant')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 64, 256])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    n_load = int(args.seconds * SR)
    t0 = time.time()
    audio, sr = sf.read(args.wav, dtype='float32', frames=n_load)
    print(f'[load] {len(audio)} samples ({len(audio)/SR:.1f}s), {sr} Hz, took {time.time()-t0:.2f}s')
    assert sr == SR, f'Expected SR={SR}, got {sr}'
    n_chunks = len(audio) // CHUNK
    audio = audio[:n_chunks * CHUNK]
    print(f'  chunks={n_chunks}')

    audio_t = torch.from_numpy(audio).to(device)

    print()
    if not args.skip_sequential:
        bench_sequential(audio_t, n_chunks, device)
        print()
    bench_offline(audio_t, device)
    print()
    bench_batched(audio_t, n_chunks, device, batch_sizes=tuple(args.batch_sizes))


if __name__ == '__main__':
    main()
