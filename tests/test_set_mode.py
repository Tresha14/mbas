"""tools/set_mode: the demo-mode switch changes only the two model lines in .env and keeps a backup."""
from tools.set_mode import describe, read_mode, set_mode

ENV = "MBAS_LAN_IP=192.168.1.33\n# a comment\nMBAS_ASR=whisper\nMBAS_WHISPER_MODEL=base.en\nMBAS_FACE=mediapipe\nMBAS_PUBLIC_URL=https://x\n"


def test_switching_to_own_changes_only_the_two_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(ENV, encoding="utf-8")
    assert describe(read_mode(env)).startswith("PRE-TRAINED ADD-ONS ON") and "whisper" in describe(read_mode(env))
    now = set_mode(env, "own")
    assert now == {"MBAS_ASR": "own", "MBAS_FACE": "own"} and describe(now).startswith("OWN")
    text = env.read_text(encoding="utf-8")
    assert "MBAS_LAN_IP=192.168.1.33" in text and "# a comment" in text and "MBAS_WHISPER_MODEL=base.en" in text and "MBAS_PUBLIC_URL=https://x" in text
    assert text.count("MBAS_ASR=") == 1 and text.count("MBAS_FACE=") == 1
    assert (tmp_path / ".env.bak").read_text(encoding="utf-8") == ENV                  # the old file is kept
    assert set_mode(env, "pretrained") == {"MBAS_ASR": "whisper", "MBAS_FACE": "mediapipe"}


def test_missing_lines_are_added_and_duplicates_collapse(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MBAS_LAN_IP=1.2.3.4\nMBAS_ASR=own\nMBAS_ASR=whisper\n", encoding="utf-8")
    assert read_mode(env)["MBAS_ASR"] == "own"                                          # the first line wins, as in the app
    set_mode(env, "pretrained")
    text = env.read_text(encoding="utf-8")
    assert text.count("MBAS_ASR=") == 1 and "MBAS_FACE=mediapipe" in text and "MBAS_LAN_IP=1.2.3.4" in text
    assert read_mode(tmp_path / "does-not-exist") == {"MBAS_ASR": "own", "MBAS_FACE": "own"}   # no .env: the defaults
