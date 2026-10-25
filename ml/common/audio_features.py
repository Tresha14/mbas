"""Audio front end: 16 kHz mono, 80-band log-mel (25 ms / 10 ms), MFCC-40, YIN pitch, RMS energy, SpecAugment.

Pure signal processing (no learned weights). The same LogMel module is used in
training and is exported inside the ONNX speech models, so features match.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000
N_FFT = 400        # 25 ms
HOP = 160          # 10 ms
N_MELS = 80
N_MFCC = 40


def load_audio(path: str | Path, sr: int = SR, start_s: float | None = None, end_s: float | None = None) -> np.ndarray:
    """float32 mono at `sr`; optional [start_s, end_s) slice read without loading the whole file."""
    info = sf.info(str(path))
    a = int(start_s * info.samplerate) if start_s is not None else 0
    b = int(end_s * info.samplerate) if end_s is not None else None
    x, file_sr = sf.read(str(path), start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        x = torchaudio.functional.resample(torch.from_numpy(x), file_sr, sr).numpy()
    return x.astype(np.float32)


class LogMel(torch.nn.Module):
    """waveform (B, T) -> log-mel (B, n_mels, frames); log(mel + 1e-6).

    The STFT is a 1-D convolution with fixed windowed cosine/sine kernels (identical to torch.stft with a periodic
    Hann window, reflect padding, power 2) so the whole front end exports to ONNX. Mel filters: HTK scale, no norm
    (torchaudio's MelSpectrogram defaults). Nothing here is learned.
    """

    def __init__(self, sr: int = SR, n_fft: int = N_FFT, hop: int = HOP, n_mels: int = N_MELS,
                 f_min: float = 20.0, f_max: float | None = None):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        win = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        ang = 2 * torch.pi * k * n / n_fft
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win])[:, None, :]       # (2F, 1, n_fft)
        self.register_buffer("basis", basis.float(), persistent=False)
        fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, f_min, f_max or sr / 2, n_mels, sr,
                                                   norm=None, mel_scale="htk")             # (F, n_mels)
        self.register_buffer("fb", fb.T.contiguous().float(), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        single = wav.dim() == 1
        x = torch.nn.functional.pad((wav[None] if single else wav)[:, None], (self.n_fft // 2, self.n_fft // 2),
                                    mode="reflect")
        spec = torch.nn.functional.conv1d(x, self.basis, stride=self.hop)                  # (B, 2F, frames)
        f = spec.shape[1] // 2
        power = spec[:, :f] ** 2 + spec[:, f:] ** 2
        out = torch.log(torch.matmul(self.fb, power) + 1e-6)
        return out[0] if single else out


_LOGMEL: LogMel | None = None


def log_mel(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(T,) or (B, T) waveform -> (80, frames) or (B, 80, frames) float32 tensor on CPU."""
    global _LOGMEL
    if _LOGMEL is None:
        _LOGMEL = LogMel()
    t = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return _LOGMEL(t)


def mfcc(x: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    import librosa

    return librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)


def pitch_yin(x: np.ndarray, sr: int = SR, fmin: float = 60.0, fmax: float = 400.0) -> np.ndarray:
    """F0 per 10 ms frame (Hz); NaN where the frame is unvoiced/silent (energy gate)."""
    import librosa

    f0 = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=HOP)
    e = rms_energy(x)[: len(f0)]
    f0 = f0[: len(e)].astype(np.float32)
    thr = max(np.percentile(e, 30), 1e-4) if len(e) else 0
    f0[e < thr] = np.nan
    return f0


def rms_energy(x: np.ndarray) -> np.ndarray:
    import librosa

    return librosa.feature.rms(y=x, frame_length=N_FFT, hop_length=HOP, center=True)[0].astype(np.float32)


def rms_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(rms_energy(x), 1e-6))


def normalize_features(f: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-utterance mean/variance normalisation over time."""
    return (f - f.mean(dim=dim, keepdim=True)) / (f.std(dim=dim, keepdim=True) + 1e-5)


class SpecAugment(torch.nn.Module):
    """Frequency + time masking on (..., n_mels, frames); only active in training mode."""

    def __init__(self, freq_masks: int = 2, freq_width: int = 27, time_masks: int = 2, time_width: float = 0.05):
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return f
        f = f.clone()
        n_mels, n_frames = f.shape[-2], f.shape[-1]
        fill = f.mean()
        for _ in range(self.freq_masks):
            w = int(torch.randint(0, self.freq_width + 1, ()).item())
            s = int(torch.randint(0, max(1, n_mels - w), ()).item())
            f[..., s:s + w, :] = fill
        tw = max(1, int(self.time_width * n_frames))
        for _ in range(self.time_masks):
            w = int(torch.randint(0, tw + 1, ()).item())
            s = int(torch.randint(0, max(1, n_frames - w), ()).item())
            f[..., :, s:s + w] = fill
        return f


def speed_perturb(x: np.ndarray, factor: float, sr: int = SR) -> np.ndarray:
    """Resample-based speed change (0.9 / 1.0 / 1.1), changes pitch like Kaldi's speed perturbation."""
    if factor == 1.0:
        return x
    y = torchaudio.functional.resample(torch.from_numpy(x), int(sr * factor), sr)
    return y.numpy().astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / max(len(noise), 1))))
    s = int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[s:s + len(x)]
    px, pn = np.mean(x ** 2) + 1e-10, np.mean(n ** 2) + 1e-10
    return (x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def synthetic_noise(n: int, rng: np.random.Generator, kind: str | None = None) -> np.ndarray:
    """Room-like noises when no recorded noise is available: white, pink, brown, fan hum, babble-like."""
    kind = kind or rng.choice(["white", "pink", "brown", "hum", "babble"])
    w = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return w
    if kind in ("pink", "brown"):
        spec = np.fft.rfft(w)
        f = np.arange(len(spec)) + 1.0
        spec /= np.sqrt(f) if kind == "pink" else f
        return np.fft.irfft(spec, n).astype(np.float32)
    t = np.arange(n) / SR
    if kind == "hum":
        base = rng.uniform(45, 120)
        return (sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 6)) / k for k in range(1, 6))
                + 0.1 * w).astype(np.float32)
    # babble-like: amplitude-modulated band noise
    env = np.abs(np.convolve(rng.standard_normal(n), np.ones(1600) / 1600, mode="same"))
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    spec[(freqs < 200) | (freqs > 3500)] = 0
    return (np.fft.irfft(spec, n) * (env / (env.max() + 1e-6))).astype(np.float32)


def speech_mask(x: np.ndarray, threshold_db: float = -35.0, hangover_frames: int = 8) -> np.ndarray:
    """Frame-level (10 ms) speech/non-speech reference from a CLEAN utterance: energy relative to its peak,
    with hangover. Used to make VAD training targets from LibriSpeech."""
    db = rms_db(x)
    on = db > (db.max() + threshold_db)
    out = on.copy()
    run = 0
    for i, v in enumerate(on):
        run = hangover_frames if v else max(run - 1, 0)
        out[i] = v or run > 0
    return out
