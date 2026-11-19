"""The interviewer report as ONE self-contained HTML file (inline styles, no scripts, no external files), for download.

It opens offline in any browser and can be emailed or archived. Everything that comes from the interview (names,
transcripts, question text) is escaped, because it is untrusted text. Built only from the stored report, so it always
agrees with the report page. It is the INTERVIEWER's report: it contains scores, so it is never sent to the candidate.
"""
from __future__ import annotations

import datetime as dt
from html import escape as e

_CSS = """
:root{--bg:#fafaf8;--card:#fff;--fg:#1c1c1a;--muted:#6b6a66;--faint:#9a9893;--line:#e7e5e0;--good:#2f7d5b;--warn:#b7791f;--bad:#b4453a;--accent:#0f766e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Inter,Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 20px 60px}h1{font:600 30px/1.2 Georgia,serif;margin:.2em 0}h2{font:600 13px/1 sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin:18px 0}.muted{color:var(--muted)}.faint{color:var(--faint)}
.top{display:flex;gap:28px;align-items:center;flex-wrap:wrap}.ring{width:120px;height:120px;border-radius:50%;display:grid;place-items:center;text-align:center;border:8px solid var(--line)}
.ring b{font:600 36px/1 Georgia,serif;display:block}.ring span{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin-top:10px}.stats b{font:600 22px Georgia,serif}.stats span{color:var(--muted);font-size:13px}
.dim{display:grid;grid-template-columns:130px 1fr 44px;gap:12px;align-items:center;margin:8px 0;font-size:14px}.bar{height:8px;border-radius:8px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:8px 6px;border-top:1px solid var(--line);vertical-align:top}th{font-weight:500;color:var(--faint);font-size:12px;border-top:0}
.b{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;background:#f0efeb;color:var(--muted)}.good{background:#e5f3ec;color:var(--good)}.watch{background:#fbf0dc;color:var(--warn)}.concern{background:#f8e4e1;color:var(--bad)}
.q{border-top:1px solid var(--line);padding:16px 0}.q:first-of-type{border-top:0}.q h3{margin:0 0 4px;font-size:16px}.tr{white-space:pre-wrap;color:var(--muted);font-size:14px;margin:8px 0}
ul{margin:6px 0 0;padding-left:20px}li{margin:3px 0}.t{display:inline-block;width:92px;color:var(--faint);font-variant-numeric:tabular-nums}
.foot{color:var(--faint);font-size:12px;margin-top:28px}@media print{body{background:#fff}.card{break-inside:avoid;border-color:#ccc}}
"""


def _n(v, d: int = 0) -> str:
    if v is None:
        return "–"
    s = f"{v:.{d}f}"
    return s.rstrip("0").rstrip(".") if d else s


def _mmss(t) -> str:
    return "" if t is None else f"{int(t // 60):02d}:{int(t % 60):02d}"


def _badge(status) -> str:
    return f"<span class='b {status}'>{e(status)}</span>" if status in ("good", "watch", "concern") else ""


def _value(r: dict) -> str:
    return "<b>–</b>" if r.get("value") is None else f"<b>{e(str(r['value']))}</b> {e(r.get('unit') or '')}"


def render_report_html(interview: dict, view: dict, generated: dt.datetime | None = None) -> str:
    sc = view.get("scorecard") or {}
    summary = view.get("summary") or {}
    overall = sc.get("overall")
    title = interview.get("title") or "Interview"
    cand = interview.get("candidate_name") or "Candidate"
    when = (generated or dt.datetime.now()).strftime("%d %B %Y, %H:%M")

    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{e(cand)} · {e(title)} · InterLens report</title><style>{_CSS}</style></head><body><main>"
           f"<p class='faint'>{e(title)} · {e(when)}</p>"]
    ring = (f"<div class='ring'><div><b>{overall:.0f}</b><span>grade {e(str(sc.get('grade') or '–'))}</span></div></div>"
            if overall is not None else "<div class='ring'><div><b>–</b><span>not rated</span></div></div>")
    verdict = (e(sc.get("grade_word") or "") if overall is not None else e(sc.get("note") or "Not enough data to rate this interview."))
    out.append(f"<div class='card top'>{ring}<div><h1>{e(cand)}</h1><div class='muted'>{verdict}</div><div class='stats'>"
               f"<div><b>{summary.get('questions', 0)}</b> <span>questions</span></div>"
               f"<div><b>{_n(summary.get('mean_effectiveness'))}</b> <span>avg. answer score</span></div>"
               f"<div><b>{_n(summary.get('mean_attention'))}</b> <span>avg. attention</span></div></div></div></div>")

    dims = sc.get("dimensions") or {}
    if dims:
        out.append("<div class='card'><h2>Scorecard</h2>")
        for k, d in dims.items():
            s = d.get("score")
            bar = f"<div class='bar'><i style='width:{max(0, min(100, s)):.0f}%'></i></div>" if s is not None else "<div class='bar'></div>"
            out.append(f"<div class='dim'><span>{e(k.replace('_', ' ').capitalize())}</span>{bar}<b>{_n(s)}</b></div>")
        if sc.get("method"):
            out.append(f"<p class='faint' style='font-size:12px'>{e(sc['method'])}</p>")
        out.append("</div>")

    dl = view.get("delivery") or {}
    if any(v is not None for k, v in dl.items() if k != "flag_counts"):
        cells = [("Pace", dl.get("wpm"), "wpm"), ("Pauses", dl.get("pause_pct"), "% of speech time"), ("Fillers", dl.get("fillers_per_100w"), "per 100 words"),
                 ("Hedges", dl.get("hedges_per_100w"), "per 100 words"), ("Camera-facing", dl.get("camera_facing_pct"), "% of the time"), ("Blinks", dl.get("blinks_per_min"), "per minute")]
        out.append("<div class='card'><h2>Delivery</h2><div class='stats'>" + "".join(
            f"<div><b>{_n(v, 1)}</b><br><span>{e(label)} · {e(unit)}</span></div>" for label, v, unit in cells) + "</div></div>")

    if view.get("moments"):
        out.append("<div class='card'><h2>Moments to rewatch</h2><ul>" + "".join(
            f"<li><span class='t'>{e(m.get('time') or '')}</span><b>{e(m.get('label') or '')}</b> <span class='muted'>{e(m.get('detail') or '')}</span></li>"
            for m in view["moments"]) + "</ul></div>")
    if view.get("combined_flags"):
        out.append("<div class='card'><h2>Where several signals dropped together</h2><ul>" + "".join(
            f"<li><span class='t'>{e(f.get('time') or '')}</span>{e(f.get('text') or '')}"
            + (f" <span class='b'>Question {f['question_order']}</span>" if f.get("question_order") is not None else "") + "</li>"
            for f in view["combined_flags"]) + "</ul></div>")

    for sec in view.get("behaviour") or []:
        rows = "".join(
            f"<tr><td>{e(r.get('label') or '')}</td><td>{_value(r)}</td><td class='faint'>{e(r.get('typical') or '')}</td><td>{_badge(r.get('status'))}</td></tr>"
            for r in sec.get("rows") or [])
        out.append(f"<div class='card'><h2>{e(sec.get('title') or '')}</h2><table><tr><th>Metric</th><th>Value</th><th>Typical</th><th></th></tr>{rows}</table></div>")

    cards = view.get("cards") or []
    if cards:
        out.append("<div class='card'><h2>Questions</h2>")
        for c in cards:
            subs = "".join(f"<span class='b'>{e(k)} {s:.0f}</span> " for k, s in (c.get("subscores") or {}).items() if s is not None)
            reasons = "".join(f"<li>{e(r.get('text') or '')}</li>" for r in (c.get("reasons") or []))
            fus = "".join(f"<li>Follow-up ({e((f.get('reason') or '').replace('_', ' '))}): {e(f.get('text') or '')}</li>" for f in (c.get("follow_ups") or []))
            eff = c.get("effectiveness")
            out.append(f"<div class='q'><h3>{e(c.get('question') or '')}</h3>"
                       f"<div class='faint'>{e(str(c.get('category') or ''))} · {_mmss(c.get('ask_s'))}"
                       f"{(' · answer score ' + str(round(eff))) if eff is not None else ''}</div><div style='margin:8px 0'>{subs}</div>"
                       f"<div class='tr'>{e(c.get('transcript') or '(no words were recognised)')}</div>"
                       f"{('<ul>' + reasons + '</ul>') if reasons else ''}{('<ul>' + fus + '</ul>') if fus else ''}</div>")
        out.append("</div>")

    out.append(f"<p class='foot'>{e(view.get('note') or 'Scores are decision support for the interviewer, never an automatic decision.')} "
               "This report is for the interviewer: it contains scores and is not to be sent to the candidate. Generated by InterLens.</p></main></body></html>")
    return "".join(out)
