"""Train the VAD from scratch, then evaluate frame F1 (raw and smoothed) on the test mixtures.

    python -m ml.speech.vad.train
    python -m ml.speech.vad.train --set train.epochs=2 data.train_items=2000      # live retrain (~2 min)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml.common.config import CHECKPOINTS, device, load_config, parse_overrides, seed_everything
from ml.common.datasets import VADDataset
from ml.common.train_utils import Run, benchmark, cosine_lr, count_params, model_card, plot_history, set_lr, write_metrics
from ml.speech.vad.model import VADNet, smooth

HERE = Path(__file__).resolve().parent


def batch_loss(model, b, dev):
    x, y = b["wav"].to(dev), b["labels"].to(dev)
    logits = model(x).float()
    n = min(logits.shape[1], y.shape[1])
    return F.binary_cross_entropy_with_logits(logits[:, :n], y[:, :n]), logits[:, :n], y[:, :n]


def load(dev="cpu") -> VADNet:
    st = torch.load(CHECKPOINTS / "vad" / "best.pt", map_location="cpu", weights_only=False)
    m = VADNet(st["cfg"]["model"]["n_mels"], st["cfg"]["model"]["ch"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval()


@torch.no_grad()
def evaluate(model, ds, dev) -> dict:
    P, Y = [], []
    for b in DataLoader(ds, batch_size=32, num_workers=4):
        _, logits, y = batch_loss(model, b, dev)
        P.append(torch.sigmoid(logits).cpu().numpy()), Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    raw = f1_score(y.ravel() > 0.5, p.ravel() > 0.5)
    sm = np.stack([smooth(r) for r in p])
    energy = []
    for b in DataLoader(ds, batch_size=32):
        x = b["wav"].numpy()
        for w in x:
            fr = w[: len(w) // 160 * 160].reshape(-1, 160)
            db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
            energy.append(np.append(db > np.percentile(db, 50), False)[: p.shape[1]])
    e = np.stack(energy)[:, : p.shape[1]]
    return {"frame_f1_raw": round(float(raw), 4), "frame_f1_smoothed": round(float(f1_score(y.ravel() > 0.5, sm.ravel())), 4),
            "energy_baseline_f1": round(float(f1_score(y.ravel() > 0.5, e.ravel())), 4),
            "speech_fraction": round(float(y.mean()), 4), "frames": int(y.size)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    tc, dc = cfg["train"], cfg["data"]
    tr = VADDataset("train", dc["seconds"], dc["train_items"])
    va = VADDataset("val", dc["seconds"], max(200, dc["train_items"] // 20))
    tr_dl = DataLoader(tr, batch_size=tc["batch_size"], shuffle=True, num_workers=dc["workers"], drop_last=True,
                       persistent_workers=dc["workers"] > 0)
    va_dl = DataLoader(va, batch_size=tc["batch_size"], num_workers=dc["workers"])
    model = VADNet(cfg["model"]["n_mels"], cfg["model"]["ch"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
    run = Run(cfg["name"], cfg)
    total = tc["epochs"] * len(tr_dl)
    run.log(f"vad: {count_params(model):,} params, {len(tr)} train mixtures of {dc['seconds']} s")
    best, step = float("inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        t = time.time()
        model.train()
        tl = 0.0
        for b in tr_dl:
            set_lr(opt, cosine_lr(step, total, tc["lr"], 200))
            loss, _, _ = batch_loss(model, b, dev)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
            step += 1
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([batch_loss(model, b, dev)[0].item() for b in va_dl]))
        is_best = vl < best
        best = min(best, vl)
        run.epoch_end(epoch, train_loss=tl / len(tr_dl), val_loss=vl)
        run.save({"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg}, best=is_best)
        run.log(f"epoch {epoch}/{tc['epochs']} train {tl / len(tr_dl):.4f} val {vl:.4f}{' *' if is_best else ''} ({time.time() - t:.0f} s)")
    run.close()

    m = load(dev)
    res = {"test": evaluate(m, VADDataset("test", dc["seconds"], 1000), dev), "params": count_params(m)}
    cpu = load()
    torch.set_num_threads(2)
    with torch.no_grad():
        res["cpu_latency_640ms_chunk"] = benchmark(lambda: cpu(torch.zeros(1, int(1.14 * 16000))))
    write_metrics("vad", res)
    plot_history("vad", ["train_loss", "val_loss"])
    model_card("vad", "Voice activity detector", {
        "Task": "Speech / non-speech every 10 ms on each participant's audio track; feeds the ASR, the pause model and "
                "the conversation analyser.",
        "Architecture": {"front end": "40-band log-mel (25 ms / 10 ms)", "network": "4 dilated 1-D conv layers (64 ch, receptive field ~330 ms)",
                         "post-processing": "hysteresis 0.5 / 0.35, 200 ms hangover, 120 ms minimum speech",
                         "parameters": f"{res['params']:,}"},
        "Training data": "LibriSpeech (+ Common Voice) utterances placed with random silences in 8 s mixtures, mixed with "
                         "recorded noise (data/raw/noise) or generated room noise (white/pink/brown/hum/babble) at 0-25 dB SNR; "
                         "targets from the clean utterances' energy mask.",
        "Results (test mixtures, unseen speakers)": {"frame F1 raw": res["test"]["frame_f1_raw"],
                                                     "frame F1 smoothed": res["test"]["frame_f1_smoothed"],
                                                     "energy-threshold baseline": res["test"]["energy_baseline_f1"],
                                                     "CPU latency per 640 ms chunk": f"{res['cpu_latency_640ms_chunk']['median_ms']} ms"},
        "Limitations": "Targets are energy masks of clean read speech; our sessions (spontaneous speech, laptop mics) are "
                       "the real test once labelled; record lab/hostel noise into data/raw/noise to replace generated noise.",
    })
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
