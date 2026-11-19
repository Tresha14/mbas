"""The downloadable HTML report: self-contained, escapes untrusted text, handles an empty report."""
import re

from backend.report_html import render_report_html
from tests.test_backend import _host, _interview, env  # noqa: F401

EVIL = "<script>alert(1)</script><img src=x onerror=alert(2)>"


def _view():
    return {
        "summary": {"questions": 1, "mean_effectiveness": 61.4, "mean_attention": 70.2},
        "scorecard": {"overall": 72.0, "grade": "B", "grade_word": "Strong", "method": "Weighted mean.",
                      "dimensions": {"content": {"score": 61.0}, "eye_contact": {"score": None}}},
        "delivery": {"wpm": 140.0, "pause_pct": 20.0, "fillers_per_100w": 1.5, "hedges_per_100w": None, "camera_facing_pct": 80.0, "blinks_per_min": 16.0},
        "moments": [{"time": "01:05", "label": "Weakest answer", "detail": EVIL}],
        "combined_flags": [{"time": "01:15–02:30", "text": f"a long pause and {EVIL} together", "question_order": 2}],
        "behaviour": [{"key": "eyes", "title": EVIL, "rows": [
            {"label": "Blink rate", "value": 16.0, "unit": "/min", "typical": "8-30", "status": "good"},
            {"label": "Face visible", "value": None, "unit": "%", "typical": "90 %", "status": "neutral"}]}],
        "cards": [{"question": EVIL, "category": "hr", "ask_s": 65.0, "effectiveness": 58.0, "subscores": {"clarity": 70.0},
                   "transcript": EVIL, "reasons": [{"text": EVIL}], "follow_ups": [{"reason": "topic_not_covered", "text": EVIL}]}],
    }


def test_html_report_is_self_contained_and_escapes_everything():
    html = render_report_html({"title": EVIL, "candidate_name": EVIL}, _view())
    assert "<script" not in html.lower() and "onerror=alert" not in html.replace("&lt;img src=x onerror=alert(2)&gt;", "")
    assert html.count("&lt;script&gt;") >= 7                                   # every place the hostile text appears is escaped
    assert not re.search(r"(src|href)=[\"']https?://", html) and "<link" not in html and "<iframe" not in html   # nothing external
    for needle in ("72", "grade B", "Blink rate", "Weakest answer", "01:15–02:30", "Question 2", "01:05", "answer score 58"):
        assert needle in html
    assert html.startswith("<!doctype html>") and "</html>" in html


def test_html_report_of_an_empty_interview_says_so_instead_of_grading():
    html = render_report_html({"title": "x", "candidate_name": None}, {"summary": {"questions": 0}, "scorecard": {
        "overall": None, "grade": None, "note": "No answers were recorded, so there is nothing to rate yet.", "dimensions": {}}})
    assert "not rated" in html and "nothing to rate yet" in html and "grade A" not in html


def test_download_endpoint_needs_ownership_and_a_built_report(env):
    c, db, bus = env
    h = _host(c)
    iv = _interview(c, h)
    assert c.get(f"/api/interviews/{iv['id']}/report.html", headers=h).status_code == 404          # not built yet
    assert c.post(f"/api/interviews/{iv['id']}/report", headers=h).status_code == 200
    r = c.get(f"/api/interviews/{iv['id']}/report.html", headers=h)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"] and "interlens-report" in r.headers["content-disposition"]
    assert c.get(f"/api/interviews/{iv['id']}/report.html").status_code in (401, 403)              # no login
    other = _host(c, "other@x.com")
    assert c.get(f"/api/interviews/{iv['id']}/report.html", headers=other).status_code == 403      # someone else's interview
