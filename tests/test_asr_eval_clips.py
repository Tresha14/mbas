from ml.speech.asr.eval_clips import score, summarise


def test_perfect_match_ignores_case_and_punctuation():
    r = score("Hello, World!", "hello world")
    assert r["errors"] == 0 and r["wer"] == 0.0


def test_counts_substitution_deletion_insertion():
    r = score("i like cats a lot", "i love cats lot")
    assert (r["sub"], r["del"], r["ins"]) == (1, 1, 0)
    r = score("hello world", "hello big world")
    assert (r["sub"], r["del"], r["ins"]) == (0, 0, 1)
    assert r["wer"] == 0.5


def test_summary_is_corpus_level_and_lists_confusions():
    rows = [score("one two three four", "one two three four"), score("a b", "a c")]
    s = summarise(rows)
    assert s["wer"] == round(1 / 6, 4)
    assert s["most_common_errors"][0] == ("b -> c", 1)
