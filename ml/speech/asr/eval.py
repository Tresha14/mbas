"""ASR evaluation: WER on LibriSpeech test-clean, Common Voice Indian-English test speakers and our test sessions
(split by accent, noise level and speaker), for greedy vs beam vs beam + our 4-gram LM.
LM weight (alpha) and word bonus (beta) are tuned on the dev set first.

    python -m ml.speech.asr.eval
    python -m ml.speech.asr.eval --limit 300          # quick check
The optional Whisper row comes from docs/metrics/whisper_benchmark.json (comparison only, never in the product;
see ml/speech/asr/benchmark_whisper.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.common import text
from ml.common.config import CHECKPOINTS, METRICS, RAW, abspath
from ml.common.datasets import ASRDataset
from ml.common.train_utils import benchmark, count_params, model_card, plot_history, write_metrics
from ml.speech.asr.decode import BeamDecoder, greedy, wer
from ml.speech.asr.model import ConformerCTC
from ml.speech.lm.ngram import LM_DIR, ArpaLM


def load(dev="cpu", which: str = "averaged") -> tuple[ConformerCTC, dict]:
    d = CHECKPOINTS / "asr"
    p = d / f"{which}.pt" if (d / f"{which}.pt").exists() else d / "best.pt"
    st = torch.load(p, map_location="cpu", weights_only=False)
    mc = st["cfg"]["model"]
    m = ConformerCTC(st["vocab"], mc["d"], mc["heads"], mc["layers"], mc["kernel"], dropout=mc["dropout"])
    m.load_state_dict(st["model"])
    return m.to(dev).eval(), st


@torch.no_grad()
def logprobs(model, ds, dev) -> list[np.ndarray]:
    out = []
    for i in range(len(ds)):
        it = ds[i]
        lp, ol = model(it["wav"][None].to(dev), torch.tensor([len(it["wav"])], device=dev))
        out.append(lp[0, : int(ol[0])].cpu().numpy())
    return out


def corpus_wer(refs, hyps) -> float:
    e = n = 0
    for r, h in zip(refs, hyps):
        a, b = wer(r, h)
        e, n = e + a, n + b
    return round(e / max(n, 1), 4)


def decode_all(lps, tok, decoder=None) -> list[str]:
    if decoder is None:
        return [tok.decode(greedy(lp)) for lp in lps]
    return [tok.decode(decoder.decode(lp)) for lp in lps]


def recorded_groups(ds: ASRDataset) -> pd.DataFrame:
    """accent / glasses / noise level per recorded test item from meta.json + session_log.csv."""
    from tools.session_log import LOG_COLUMNS  # noqa: F401  (session log format documented there)

    rows = []
    log = pd.read_csv(RAW / "session_log.csv") if (RAW / "session_log.csv").exists() else pd.DataFrame()
    for r in ds.m.itertuples():
        if r.source != "recorded":
            continue
        sid = r.item_id.rsplit("_A", 1)[0]
        meta = json.loads((RAW / sid / "meta.json").read_text(encoding="utf-8"))
        c = meta["participants"]["candidate"]
        noise = log.loc[log.session_id == sid, "noise"].iloc[0] if len(log) and (log.session_id == sid).any() else "unknown"
        rows.append({"item_id": r.item_id, "accent": c.get("accent"), "speaker": c.get("participant_id"), "noise": noise})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--beam", type=int, default=16)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, st = load(dev)
    tok = text.load("asr")
    pieces = tok.pieces()
    lm_p = LM_DIR / "interview_4gram.arpa"
    lm = ArpaLM(lm_p) if lm_p.exists() else None

    # tune alpha / beta on dev
    dev_ds = ASRDataset("val", tok, ("librispeech",), 20.0, limit=min(args.limit or 300, 300))
    dev_lp = logprobs(model, dev_ds, dev)
    refs = dev_ds.m.norm.tolist()
    best = (corpus_wer(refs, decode_all(dev_lp, tok, BeamDecoder(pieces, None, beam=args.beam))), 0.0, 0.0)
    tuning = []
    if lm:
        for a in (0.3, 0.5, 0.8):
            for b in (0.0, 1.0, 2.0):
                w = corpus_wer(refs, decode_all(dev_lp, tok, BeamDecoder(pieces, lm, a, b, beam=args.beam)))
                tuning.append({"alpha": a, "beta": b, "dev_wer": w})
                if w < best[0]:
                    best = (w, a, b)
    alpha, beta = best[1], best[2]

    sets = {"librispeech_test_clean": ASRDataset("test", tok, ("librispeech",), 35.0, limit=args.limit),
            "common_voice_indian_test": ASRDataset("test", tok, ("common_voice_in",), 35.0, limit=args.limit)
            if (abspath("data/processed/public/common_voice_in/manifest.parquet")).exists() else None,
            "our_sessions_test": ASRDataset("test", tok, ("recorded",), 60.0, limit=args.limit)}
    table = {}
    for name, ds in sets.items():
        if ds is None or len(ds) == 0:
            table[name] = {"note": "no data yet"}
            continue
        lps = logprobs(model, ds, dev)
        refs = ds.m.norm.tolist()
        hyp_g = decode_all(lps, tok)
        hyp_b = decode_all(lps, tok, BeamDecoder(pieces, None, beam=args.beam))
        hyp_l = decode_all(lps, tok, BeamDecoder(pieces, lm, alpha, beta, beam=args.beam)) if lm else hyp_b
        row = {"utterances": len(ds), "hours": round(float(ds.durations().sum() / 3600), 2), "greedy": corpus_wer(refs, hyp_g),
               "beam": corpus_wer(refs, hyp_b), "beam_lm": corpus_wer(refs, hyp_l),
               "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyp_l))[:5]]}
        if name == "our_sessions_test":
            g = recorded_groups(ds)
            for col in ("accent", "noise", "speaker"):
                row[f"by_{col}"] = {k: corpus_wer([refs[i] for i in idx], [hyp_l[i] for i in idx])
                                    for k, idx in g.groupby(col).indices.items()}
        table[name] = row
    whisper = json.loads((METRICS / "whisper_benchmark.json").read_text()) if (METRICS / "whisper_benchmark.json").exists() else None
    cpu, _ = load("cpu")
    torch.set_num_threads(4)
    x = torch.zeros(1, int(4.96 * 16000))
    with torch.no_grad():
        lat = benchmark(lambda: cpu(x, torch.tensor([x.shape[1]])), warmup=2, iters=10)
    res = {"checkpoint": st.get("averaged_from", "best"), "epochs_trained": st.get("epoch"), "params": count_params(model),
           "lm": str(lm_p.name) if lm else None, "alpha": alpha, "beta": beta, "lm_tuning_dev": tuning, "wer": table,
           "whisper_small_benchmark": whisper or "not run (optional, comparison only)",
           "cpu_latency_window_4.96s_ms": lat}
    write_metrics("asr", res)
    plot_history("asr", ["train_ctc", "val_wer_greedy"])
    t = table["librispeech_test_clean"]
    model_card("asr", "Conformer-CTC speech recogniser", {
        "Task": "Streaming transcription of each participant's audio track (< 2 s delay); word timestamps for the "
                "transcript panel, the pause model and the fusion features.",
        "Architecture": {"front end": "80-band log-mel, per-utterance CMVN, SpecAugment (2 freq x 27, 10 time x 5%)",
                         "encoder": f"Conv2d x2 subsampling (40 ms) + {st['cfg']['model']['layers']} Conformer blocks, d_model "
                                    f"{st['cfg']['model']['d']}, {st['cfg']['model']['heads']} heads, conv kernel 31",
                         "output": f"CTC over our SentencePiece vocabulary ({st['vocab']} pieces)", "parameters": f"{res['params']:,}"},
        "Training": "LibriSpeech train-clean-100 (+ Common Voice Indian English + our transcripts when available); speed "
                    "perturbation 0.9/1.0/1.1, Noam schedule, fp16, dynamic chunk training (streaming), checkpoint averaging.",
        "Language model": f"our interpolated Kneser-Ney 4-gram (ml/speech/lm/ngram.py), alpha {alpha}, beta {beta}",
        "WER (greedy / beam / beam+LM)": {k: (f"{v['greedy']} / {v['beam']} / {v['beam_lm']}" if "greedy" in v else v["note"])
                                          for k, v in table.items()},
        "Whisper-small (benchmark only, not in the product)": str(whisper or "not run"),
        "Why a gap to Whisper is expected": "Whisper-small was trained on 680,000 h of audio; ours on ~100 h. The gap "
                                            "comes from training data size, not from the architecture.",
    })
    print(json.dumps({k: v for k, v in res.items() if k != "lm_tuning_dev"}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
