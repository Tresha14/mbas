"""Train the fusion network + ablation (video / audio / text only / all) + ridge baseline + explanations.

    python -m ml.fusion.answer_features        # first: out-of-fold answer-model features
    python -m ml.fusion.train
Writes checkpoints/fusion/best.pt, docs/metrics/fusion.json, docs/metrics/fusion_explanations.json,
docs/model_cards/fusion.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml.common.config import CHECKPOINTS, METRICS, device, load_config, parse_overrides, save_json, seed_everything
from ml.common.datasets import FOLLOW_UP_REASONS, FUSION_FEATURES, FUSION_TARGETS, FusionDataset, fusion_table
from ml.common.train_utils import Run, count_params, model_card, plot_history, write_metrics
from ml.fusion.explain import top_reasons
from ml.fusion.model import MODALITIES, FusionNet

HERE = Path(__file__).resolve().parent
ABLATIONS = {"video only": ["vision"], "audio only": ["prosody", "hesitation"], "text only": ["content"],
             "all modalities": MODALITIES}


def loss_fn(o, b, tc):
    l_s = F.mse_loss(o["scores"], b["scores"]) * 10
    l_f = F.binary_cross_entropy_with_logits(o["follow_up"], b["follow_up"])
    l_r = F.cross_entropy(o["reason"], b["reason"])
    return l_s + tc["follow_up_weight"] * l_f + tc["reason_weight"] * l_r


def train_one(cfg, tr, va, dev, active, run: Run | None = None, seed: int = 0):
    torch.manual_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    model = FusionNet(d=mc["d"], heads=mc["heads"], dropout=mc["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc["epochs"])
    tr_dl = DataLoader(tr, batch_size=tc["batch_size"], shuffle=True, drop_last=True)
    va_dl = DataLoader(va, batch_size=512)
    best, best_state = float("inf"), None
    rng = np.random.default_rng(seed)
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        tl = 0.0
        for b in tr_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            act = [m for m in active if rng.random() > tc["modality_dropout"]] or active
            o = model(b, act)
            loss = loss_fn(o, b, tc)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_fn(model({k: v.to(dev) for k, v in b.items()}, active), {k: v.to(dev) for k, v in b.items()}, tc).item()
                          for b in va_dl])
        if vl < best:
            best, best_state = vl, {k: t.detach().clone() for k, t in model.state_dict().items()}
        if run:
            run.epoch_end(epoch, train_loss=tl / len(tr_dl), val_loss=vl)
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def evaluate(model, ds, dev, active) -> dict:
    b = {k: torch.from_numpy(np.stack([ds[i][k].numpy() for i in range(len(ds))])).to(dev) for k in ds[0]}
    o = model(b, active)
    s = o["scores"].cpu().numpy() * 100
    y = b["scores"].cpu().numpy() * 100
    fu_p = torch.sigmoid(o["follow_up"]).cpu().numpy()
    fu = b["follow_up"].cpu().numpy()
    reason = o["reason"].argmax(1).cpu().numpy()
    res = {"effectiveness_mae": round(float(np.abs(s[:, 0] - y[:, 0]).mean()), 3),
           "effectiveness_r": round(float(np.corrcoef(s[:, 0], y[:, 0])[0, 1]), 4),
           "subscore_mae": {t: round(float(np.abs(s[:, i] - y[:, i]).mean()), 3) for i, t in enumerate(FUSION_TARGETS[1:], 1)},
           "follow_up_f1": round(float(f1_score(fu, fu_p > 0.5)), 4),
           "reason_macro_f1": round(float(f1_score(b["reason"].cpu().numpy(), reason, average="macro",
                                                   labels=range(len(FOLLOW_UP_REASONS)))), 4),
           "modality_weights": dict(zip(MODALITIES, np.round(o["modality_weights"].mean(0).cpu().numpy(), 4).tolist()))}
    return res


def flat(ds, i) -> np.ndarray:
    it = ds[i]
    return np.concatenate([it[m].numpy() for m in FUSION_FEATURES])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg["seed"])
    dev = device()
    table = fusion_table(cfg["source"])
    tr = FusionDataset("train", table=table)
    va, te = FusionDataset("val", table=table, stats=tr.stats), FusionDataset("test", table=table, stats=tr.stats)
    have_ans = table["ans_relevance"].notna().mean()
    run = Run(cfg["name"], cfg)
    run.log(f"fusion: {len(tr)} / {len(va)} / {len(te)} answers; answer-model features present for {have_ans:.0%}")

    ablation = {}
    models = {}
    for name, act in ABLATIONS.items():
        m = train_one(cfg, tr, va, dev, act, run if name == "all modalities" else None)
        ablation[name] = evaluate(m, te, dev, act)
        models[name] = m
        run.log(f"{name:15s} effectiveness MAE {ablation[name]['effectiveness_mae']:.2f}  follow-up F1 {ablation[name]['follow_up_f1']:.3f}")
    main_model = models["all modalities"].cpu()

    # ridge baseline on all standardised features
    X = lambda d: np.concatenate([d.x[m] for m in FUSION_FEATURES], 1)  # noqa: E731
    ridge = Ridge(alpha=10.0).fit(X(tr), tr.y[:, 0] * 100)
    rp = ridge.predict(X(te))
    baseline = {"ridge_effectiveness_mae": round(float(np.abs(rp - te.y[:, 0] * 100).mean()), 3),
                "mean_effectiveness_mae": round(float(np.abs(tr.y[:, 0].mean() * 100 - te.y[:, 0] * 100).mean()), 3)}

    # explanations: top-3 reasons for test answers
    ex = []
    raw_cols = [c for cols in FUSION_FEATURES.values() for c in cols]
    for i in range(min(cfg["explain"]["examples"], len(te))):
        raw = {c: (None if np.isnan(v) else float(v)) for c, v in zip(raw_cols, te.t.loc[i, raw_cols].astype(float))}
        x = flat(te, i)
        with torch.no_grad():
            o = main_model({m: torch.from_numpy(te.x[m][i:i + 1]) for m in FUSION_FEATURES})
        ex.append({"answer_id": te.t.answer_id.iloc[i], "effectiveness": round(float(o["scores"][0, 0]) * 100, 1),
                   "true_effectiveness": float(te.t.communication_effectiveness.iloc[i]),
                   "follow_up_p": round(float(torch.sigmoid(o["follow_up"][0])), 3),
                   "reason": FOLLOW_UP_REASONS[int(o["reason"][0].argmax())],
                   "modality_weights": dict(zip(MODALITIES, np.round(o["modality_weights"][0].numpy(), 3).tolist())),
                   "top_reasons": top_reasons(main_model, x, raw, 3, "effectiveness")})
    save_json(ex, METRICS / "fusion_explanations.json")
    coverage = float(np.mean([len(e["top_reasons"]) == 3 for e in ex]))

    ck = CHECKPOINTS / "fusion"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save({"model": main_model.state_dict(), "cfg": cfg,
                "stats": {m: (mu.tolist(), sd.tolist()) for m, (mu, sd) in tr.stats.items()},
                "features": FUSION_FEATURES}, ck / "best.pt")
    res = {"source": cfg["source"], "answers": {"train": len(tr), "val": len(va), "test": len(te)},
           "answer_model_feature_coverage": round(float(have_ans), 4), "ablation": ablation, "baselines": baseline,
           "explanation_coverage_top3": coverage, "params": count_params(main_model)}
    write_metrics("fusion", res)
    plot_history(cfg["name"], ["train_loss", "val_loss"])
    run.close()
    a = ablation["all modalities"]
    model_card("fusion", "Multimodal fusion network", {
        "Task": "Communication effectiveness 0-100 with clarity, confidence, engagement, relevance and fluency "
                "sub-scores, follow-up yes/no + reason, with the top-3 reasons for every score.",
        "Architecture": {"encoders": "one MLP per modality (vision, prosody, hesitation, content, conversation) -> 64-d token",
                         "fusion": "learned query, 4-head attention over modality tokens (weights = modality importance) + mean pool",
                         "heads": "6 sigmoid scores, follow-up logit, 6-way reason", "parameters": f"{res['params']:,}",
                         "regularisation": f"dropout {cfg['model']['dropout']}, modality dropout {cfg['train']['modality_dropout']}"},
        "Data": f"{cfg['source']} answers, person-wise split; content modality = out-of-fold answer-model outputs. "
                "SYNTHETIC for now: retrain on MIT Interview ratings + our labels with the same script.",
        "Ablation (test, effectiveness MAE / follow-up F1)": {k: f"{v['effectiveness_mae']} / {v['follow_up_f1']}" for k, v in ablation.items()},
        "Baselines": baseline,
        "All modalities (test)": {"effectiveness MAE": a["effectiveness_mae"], "Pearson r": a["effectiveness_r"],
                                  "follow-up F1": a["follow_up_f1"], "reason macro-F1": a["reason_macro_f1"],
                                  "mean modality attention": a["modality_weights"]},
        "Explanations": f"Shapley values by permutation sampling ({cfg['explain']['n_perm']} orderings) -> templates "
                        f"(ml/fusion/explain.py); {coverage:.0%} of test answers get 3 reasons "
                        "(examples: docs/metrics/fusion_explanations.json).",
        "Use": "Decision support for the interviewer, never an automatic reject.",
    })
    print({k: res[k] for k in ("ablation", "baselines", "explanation_coverage_top3")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
