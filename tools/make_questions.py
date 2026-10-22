"""Build recorder/questions.json (40 questions).

Q01-Q36 come from the synthetic dataset's reference/question_bank.csv so IDs
match the synthetic data; Q37-Q40 are extra questions added for recording.

    python -m tools.make_questions
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "raw" / "mbas_synthetic_dataset" / "reference" / "question_bank.csv"
OUT = ROOT / "recorder" / "questions.json"

EXTRA = [
    {
        "question_id": "Q37",
        "category": "behavioural",
        "difficulty": 0.5,
        "text": "Tell me about a time you had to learn something new very quickly to finish a task.",
        "key_points": [
            "Situation: the deadline and what was unfamiliar.",
            "Task: what had to be delivered.",
            "Action: how the topic was learned (docs, small experiments, asking for help).",
            "Result: the outcome and what was learned about learning.",
        ],
    },
    {
        "question_id": "Q38",
        "category": "behavioural",
        "difficulty": 0.5,
        "text": "Describe a situation where you received critical feedback. How did you respond?",
        "key_points": [
            "Situation: who gave the feedback and on what.",
            "Task: what needed to change.",
            "Action: how the feedback was accepted and acted on.",
            "Result: the improvement that followed.",
        ],
    },
    {
        "question_id": "Q39",
        "category": "technical",
        "difficulty": 0.55,
        "text": "What is the difference between a process and a thread?",
        "key_points": [
            "A process has its own address space; threads share the address space of their process.",
            "Creating and switching threads is cheaper than processes.",
            "Threads need synchronisation (locks) for shared data; processes communicate through IPC.",
            "A crash in one thread can bring down the whole process.",
        ],
    },
    {
        "question_id": "Q40",
        "category": "technical",
        "difficulty": 0.6,
        "text": "How would you design a URL shortener?",
        "key_points": [
            "Generate a short unique key (base62 counter or hash) and store key -> URL.",
            "Redirect with HTTP 301/302 on lookup.",
            "Use a cache for hot keys and a database for persistence.",
            "Handle collisions, expiry, and rate limiting.",
        ],
    },
]


def main() -> None:
    questions = []
    with BANK.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "difficulty": float(row["difficulty"]),
                    "text": row["text"],
                    "key_points": [row[f"key_point_{i}"] for i in range(1, 5) if row.get(f"key_point_{i}")],
                }
            )
    questions += EXTRA
    assert len(questions) == 40, len(questions)
    OUT.write_text(json.dumps({"version": 1, "questions": questions}, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    print(f"wrote {OUT} ({len(questions)} questions: {counts})")


if __name__ == "__main__":
    main()
