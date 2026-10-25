"""docs/model_report.md: every model - data, architecture, parameters, epochs, curves, test metrics, confusion
matrices, failure cases - assembled from docs/metrics/*.json, docs/model_cards/*.md and runs/*/history.json.

    python -m tools.model_report
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics"
CARDS = ROOT / "docs" / "model_cards"
RUNS = ROOT / "runs"

MODELS = [  # (name, stage, title, headline metric getter)
    ("face_detector", 4, "Face + landmark detector", lambda m: f"webcam-crop AP@0.5 {m['webcam_crops']['AP@0.5']}, "
                                                              f"recall {list(m['webcam_crops'].values())[3]}"),
    ("gaze_iris", 4, "Gaze + iris CNN", lambda m: f"angular error {m['angular_error_mean_deg']} deg (baseline {m['baseline_mean_gaze_error_deg']})"),
    ("headpose", 5, "Head-pose CNN", lambda m: f"MAE yaw/pitch/roll {m['test_mae_deg']['yaw']}/{m['test_mae_deg']['pitch']}/{m['test_mae_deg']['roll']} deg"),
    ("window_models", 5, "Engagement MLP + attention score", lambda m: f"engagement macro-F1 {m['engagement_window_mlp']['macro_f1']}, "
                                                                    f"attention MAE {m['attention'][m['attention_selected']]['score_mae']}"),
    ("engagement", 5, "Engagement CNN-GRU", lambda m: "; ".join(f"{r['setting']}: {r['macro_f1']}" for r in m["comparison"])),
    ("vad", 6, "Voice activity detector", lambda m: f"frame F1 {m['test']['frame_f1_smoothed']} (energy baseline {m['test']['energy_baseline_f1']})"),
    ("asr", 6, "Conformer-CTC ASR", lambda m: "; ".join(f"{k}: {v.get('greedy')}/{v.get('beam')}/{v.get('beam_lm')}"
                                                         for k, v in m["wer"].items() if "greedy" in v) + " (WER greedy/beam/beam+LM)"),
    ("disfluency", 7, "Disfluency CRNN", lambda m: ", ".join(f"{k} F1 {v['f1']}" for k, v in m["per_class"].items())),
    ("behaviour", 7, "Speaking pattern + hesitation index", lambda m: f"pattern macro-F1 {m['speaking_pattern']['macro_f1']}, "
                                                                     f"hesitation MAE {m['hesitation_index']['weighted_sum_mae']}"),
    ("answer_quality", 8, "Answer-quality transformer", lambda m: f"quality QWK {m['transformer']['quality_qwk']} vs TF-IDF {m['tfidf_baseline']['quality_qwk']}"),
    ("fusion", 8, "Multimodal fusion", lambda m: f"effectiveness MAE {m['ablation']['all modalities']['effectiveness_mae']}, "
                                                 f"follow-up F1 {m['ablation']['all modalities']['follow_up_f1']}"),
    ("meeting_head", 8, "Meeting head", lambda m: f"AMI balance MAE {m['relative_turn_features (used)'].get('ami', {}).get('balance_mae_vs_formula')}"),
]


def epochs(name: str) -> int | None:
    h = RUNS / name / "history.json"
    return len(json.loads(h.read_text())) if h.exists() else None


def main(argv: list[str] | None = None) -> int:
    lines = ["# MBAS model report", "", f"_Generated {dt.date.today().isoformat()} by `python -m tools.model_report`._", "",
             "Every model was defined in PyTorch and trained from random initialisation on public raw datasets, the "
             "synthetic dataset and (when labelled) our own recordings. No pre-trained weights or AI APIs are used.", "",
             "## Summary", "", "| Stage | Model | Epochs | Headline test result |", "| --- | --- | ---: | --- |"]
    details = []
    for name, stage, title, head in MODELS:
        mp = METRICS / f"{name}.json"
        if not mp.exists():
            lines.append(f"| {stage} | {title} | - | not trained yet |")
            continue
        m = json.loads(mp.read_text())
        try:
            h = head(m)
        except (KeyError, TypeError, IndexError):
            h = "see metrics JSON"
        lines.append(f"| {stage} | {title} | {epochs(name) or '-'} | {h} |")
        card = CARDS / f"{name}.md"
        details.append(f"\n---\n\n## Stage {stage}: {title}\n")
        if card.exists():
            details.append("\n".join(card.read_text(encoding="utf-8").splitlines()[3:]))
        for img in sorted(METRICS.glob(f"{name}*.png")) + sorted(METRICS.glob(f"*{name.split('_')[0]}*_cm*.png")):
            details.append(f"\n![{img.stem}](metrics/{img.name})")
        details.append(f"\nMetrics: `docs/metrics/{name}.json`")
    lat = METRICS / "onnx_latency.md"
    if lat.exists():
        details.append("\n---\n\n" + lat.read_text(encoding="utf-8").replace("# ONNX", "## ONNX"))
    details.append("\n---\n\n## Failure cases and limitations\n")
    details += ["- Webcam vision models are pre-trained on public photos (WIDER, MPIIGaze); dim and backlit webcams and "
                "strong glasses reflections are the main failure cases until our recordings are added.",
                "- The answer, behaviour and fusion models are trained on the SYNTHETIC dataset; their scores show that the "
                "pipeline learns, not how well it rates real candidates. Re-train on our labelled sessions with the same scripts.",
                "- The ASR is trained on ~100 h of read English; spontaneous Indian-English interview speech is harder "
                "(see the per-accent WER rows once our sessions are transcribed).",
                "- Bias check across gender, accent and glasses: docs/bias_check.md."]
    out = ROOT / "docs" / "model_report.md"
    out.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
