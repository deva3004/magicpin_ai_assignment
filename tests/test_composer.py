"""Tests every deterministic and network-free part of composer.py: the
validation pipeline, the fallback path, memoization, context trimming, and
that send_as/suppression_key are mechanically derived rather than trusted
from the LLM's own output. The real LLM call is monkeypatched out throughout
— these must run without an API key or network access.
"""

import json

import pytest

from bot import composer
from bot.grounding import Claim
from bot.voice import load_voice

CATEGORY = {
    "slug": "dentists",
    "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
    "digest": [{"id": "d1", "title": "t", "summary": "38% better outcomes", "trial_n": 2100}],
}


@pytest.fixture(autouse=True)
def clear_caches():
    composer._CACHE.clear()
    composer._REPLY_CACHE.clear()
    yield
    composer._CACHE.clear()
    composer._REPLY_CACHE.clear()


# ---------------------------------------------------------------------------
# _passes_validation
# ---------------------------------------------------------------------------

def _bundle():
    return {"category": CATEGORY, "merchant": {}, "trigger": {}, "customer": None}


def test_passes_validation_rejects_taboo_word():
    voice = load_voice(CATEGORY)
    assert not composer._passes_validation("This is guaranteed to work.", "none", [], voice, _bundle())


def test_passes_validation_rejects_url():
    voice = load_voice(CATEGORY)
    assert not composer._passes_validation("Check https://example.com for more.", "none", [], voice, _bundle())


def test_passes_validation_rejects_internal_id_leak():
    voice = load_voice(CATEGORY)
    body = "See d_2026W17_jida_fluoride for details."
    assert not composer._passes_validation(body, "none", [], voice, _bundle())


def test_passes_validation_rejects_multiple_question_marks():
    voice = load_voice(CATEGORY)
    assert not composer._passes_validation("Reply YES? Or reply NO?", "binary_yes_no", [], voice, _bundle())


def test_passes_validation_allows_zero_question_marks_for_imperative_cta():
    # Case Study 2's real gold CTA shape — imperative, no "?" at all.
    voice = load_voice(CATEGORY)
    assert composer._passes_validation("Reply 1 for Wed, 2 for Thu.", "multi_choice_slot", [], voice, _bundle())


def test_passes_validation_rejects_question_mark_when_cta_is_none():
    voice = load_voice(CATEGORY)
    assert not composer._passes_validation("Is this ok?", "none", [], voice, _bundle())


def test_passes_validation_rejects_unresolved_claim():
    voice = load_voice(CATEGORY)
    claims = [Claim(text="fake", source_path="category.nonexistent")]
    assert not composer._passes_validation("Some text.", "none", claims, voice, _bundle())


def test_passes_validation_accepts_resolved_claim():
    voice = load_voice(CATEGORY)
    claims = [Claim(text="38%", source_path="category.digest[0].summary")]
    assert composer._passes_validation("Results show 38% better outcomes.", "none", claims, voice, _bundle())


# ---------------------------------------------------------------------------
# _build_bundle trimming
# ---------------------------------------------------------------------------

def test_build_bundle_trims_digest_to_referenced_item():
    category = {
        "slug": "dentists",
        "digest": [{"id": "d1", "title": "one"}, {"id": "d2", "title": "two"}],
        "patient_content_library": [{"id": "pc1"}],
    }
    trigger = {"payload": {"top_item_id": "d2"}}
    bundle = composer._build_bundle(category, {}, trigger, None)
    assert bundle["category"]["digest"] == [{"id": "d2", "title": "two"}]
    assert "patient_content_library" not in bundle["category"]


def test_build_bundle_keeps_full_digest_when_no_top_item_id():
    category = {"digest": [{"id": "d1"}, {"id": "d2"}]}
    bundle = composer._build_bundle(category, {}, {"payload": {}}, None)
    assert len(bundle["category"]["digest"]) == 2


def test_build_bundle_caps_conversation_history_to_last_five():
    merchant = {"conversation_history": [{"from": "vera", "body": f"msg{i}"} for i in range(10)]}
    bundle = composer._build_bundle({}, merchant, {"payload": {}}, None)
    history = bundle["merchant"]["conversation_history"]
    assert len(history) == 5
    assert history[0]["body"] == "msg5"


# ---------------------------------------------------------------------------
# compose() — fallback, memoization, mechanical send_as/suppression_key
# ---------------------------------------------------------------------------

def _merchant():
    return {"merchant_id": "m_001", "category_slug": "dentists",
            "identity": {"name": "Test Clinic", "owner_first_name": "Sam"}}


def _trigger(kind="research_digest"):
    return {"id": "trg_1", "kind": kind, "suppression_key": "sk1", "payload": {}}


def test_compose_falls_back_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: None)
    result = composer.compose(CATEGORY, _merchant(), _trigger())
    assert "fallback" in result.rationale.lower()
    assert result.send_as == "vera"
    assert result.suppression_key == "sk1"


def test_compose_is_memoized_for_identical_inputs(monkeypatch):
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: None)
    r1 = composer.compose(CATEGORY, _merchant(), _trigger())
    r2 = composer.compose(CATEGORY, _merchant(), _trigger())
    assert r1 is r2


def test_compose_uses_valid_llm_output_over_fallback(monkeypatch):
    fake = json.dumps({
        "body": "Test Clinic, quick heads up on this week's digest item. Want details?",
        "cta": "open_ended", "rationale": "Testing.", "claims": [],
    })
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: fake)
    result = composer.compose(CATEGORY, _merchant(), _trigger())
    assert "fallback" not in result.rationale.lower()
    assert result.body.startswith("Test Clinic")


def test_compose_send_as_is_mechanical_regardless_of_llm_content(monkeypatch):
    fake = json.dumps({"body": "Hi there, quick question.", "cta": "none", "rationale": "x", "claims": []})
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: fake)
    customer = {"customer_id": "c_1", "identity": {"name": "Priya"}}
    result = composer.compose(CATEGORY, _merchant(), _trigger("recall_due"), customer)
    assert result.send_as == "merchant_on_behalf"


def test_compose_falls_back_when_llm_output_fails_validation(monkeypatch):
    # LLM claims a fact that doesn't resolve anywhere in the bundle
    fake = json.dumps({
        "body": "We saw a 71% jump in bookings!", "cta": "none", "rationale": "x",
        "claims": [{"text": "71%", "source_path": "category.digest[0].trial_n"}],
    })
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: fake)
    result = composer.compose(CATEGORY, _merchant(), _trigger())
    assert "fallback" in result.rationale.lower()


def test_compose_falls_back_on_malformed_json(monkeypatch):
    monkeypatch.setattr(composer, "_call_llm", lambda *a, **kw: "not json at all {{{")
    result = composer.compose(CATEGORY, _merchant(), _trigger())
    assert "fallback" in result.rationale.lower()
