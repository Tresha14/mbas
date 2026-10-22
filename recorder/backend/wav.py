"""Incrementally written PCM WAV files.

The header is rewritten after every append, so the file on disk is always a
valid WAV even if the recording is interrupted.
"""
from __future__ import annotations

import struct
from pathlib import Path

HEADER_SIZE = 44


def _header(data_bytes: int, sample_rate: int, channels: int, bits: int) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_bytes)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_bytes)
    )


def create(path: Path, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> None:
    path.write_bytes(_header(0, sample_rate, channels, bits))


def append(path: Path, pcm: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> int:
    """Append raw little-endian PCM and fix the header. Returns total data bytes."""
    with path.open("r+b") as f:
        f.seek(0, 2)
        f.write(pcm)
        data_bytes = f.tell() - HEADER_SIZE
        f.seek(0)
        f.write(_header(data_bytes, sample_rate, channels, bits))
    return data_bytes


def info(path: Path) -> dict:
    """Read sample rate, channels, bit depth and duration from a PCM WAV."""
    with path.open("rb") as f:
        riff = f.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path} is not a WAV file")
        fmt = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path}: no data chunk")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16, 1)
            elif cid == b"data":
                if fmt is None:
                    raise ValueError(f"{path}: data before fmt")
                _, channels, sr, _, block_align, bits = fmt
                frames = size // block_align
                return {
                    "sample_rate": sr,
                    "channels": channels,
                    "bits": bits,
                    "frames": frames,
                    "duration_s": frames / sr,
                }
            else:
                f.seek(size + (size & 1), 1)
