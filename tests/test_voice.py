import json
from pathlib import Path

from bot.voice import count_ctas, find_taboo_violations, has_buried_cta, load_voice

CATEGORY = json.loads(Path("dataset/categories/dentists.json").read_text())


def test_load_voice_parses_real_category():
    voice = load_voice(CATEGORY)
    assert voice.tone == "peer_clinical"
    assert "guaranteed" in voice.vocab_taboo


def test_taboo_violation_plain_word():
    voice = load_voice(CATEGORY)
    assert "guaranteed" in find_taboo_violations("This is guaranteed to work.", voice)
    assert find_taboo_violations("This is a routine cleaning.", voice) == []


def test_taboo_violation_strips_parenthetical_qualifier():
    # dentists.json's taboo list has "FDA-approved (use only when actually
    # applicable)" baked in as one literal string — matched literally that
    # phrase would never appear in real prose, silently disabling the check.
    voice = load_voice(CATEGORY)
    hits = find_taboo_violations("Our treatment is FDA-approved.", voice)
    assert hits == ["FDA-approved (use only when actually applicable)"]


def test_count_ctas_counts_question_marks():
    assert count_ctas("No question here.") == 0
    assert count_ctas("Want me to help?") == 1
    assert count_ctas("Reply YES for X? Or reply NO for Y?") == 2


def test_gold_case_study_2_cta_has_zero_question_marks():
    # Real gold CTA from case-studies.md — imperative, not a question.
    # Documents why composer.py never requires exactly one "?".
    body = "Reply 1 for Wed, 2 for Thu, or tell us a time that works."
    assert count_ctas(body) == 0


def test_has_buried_cta_true_for_real_trailing_content():
    body = "Want me to help? Also here's a bunch of unrelated extra info that keeps going on and on well past fifty characters."
    assert has_buried_cta(body)


def test_has_buried_cta_false_for_trailing_citation():
    # Several gold examples legitimately have a short source citation after
    # the question mark — must not be flagged as "buried".
    body = "Want me to pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14"
    assert not has_buried_cta(body)


def test_has_buried_cta_false_when_no_question_mark():
    assert not has_buried_cta("Just an FYI, no ask here.")
