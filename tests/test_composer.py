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
    composer._rate_limit_state["remaining_tokens"] = None
    composer._rate_limit_state["resets_at"] = None
    yield
    composer._CACHE.clear()
    composer._REPLY_CACHE.clear()
    composer._rate_limit_state["remaining_tokens"] = None
    composer._rate_limit_state["resets_at"] = None


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


# ---------------------------------------------------------------------------
# Groq free-tier rate-limit tracking — reasoning_effort + preemptive skip.
# Verified live against Groq (see problemFaced.txt): reasoning_effort="low"
# is a real accepted param that cuts hidden chain-of-thought tokens, and
# Groq returns x-ratelimit-remaining-tokens/-reset-tokens on every response.
# ---------------------------------------------------------------------------

def test_parse_reset_seconds_plain():
    assert composer._parse_reset_seconds("14.309s") == pytest.approx(14.309)


def test_parse_reset_seconds_with_minutes():
    assert composer._parse_reset_seconds("2m3.5s") == pytest.approx(123.5)


def test_rate_limit_not_exhausted_when_no_state_recorded():
    assert not composer._rate_limit_likely_exhausted(1000)


def test_rate_limit_exhausted_when_remaining_below_estimate():
    composer._rate_limit_state["remaining_tokens"] = 100.0
    composer._rate_limit_state["resets_at"] = composer.time.monotonic() + 30
    assert composer._rate_limit_likely_exhausted(1000)


def test_rate_limit_not_exhausted_once_reset_window_has_passed():
    composer._rate_limit_state["remaining_tokens"] = 100.0
    composer._rate_limit_state["resets_at"] = composer.time.monotonic() - 1  # already elapsed
    assert not composer._rate_limit_likely_exhausted(1000)


def test_update_rate_limit_state_reads_groq_headers():
    class FakeResp:
        headers = {"x-ratelimit-remaining-tokens": "4200", "x-ratelimit-reset-tokens": "10s"}

    composer._update_rate_limit_state(FakeResp())
    assert composer._rate_limit_state["remaining_tokens"] == 4200.0
    assert composer._rate_limit_state["resets_at"] > composer.time.monotonic()


def test_update_rate_limit_state_ignores_response_without_headers():
    class FakeResp:
        headers: dict = {}

    composer._update_rate_limit_state(FakeResp())
    assert composer._rate_limit_state["remaining_tokens"] is None


def test_call_llm_skips_network_when_groq_budget_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    composer._rate_limit_state["remaining_tokens"] = 10.0
    composer._rate_limit_state["resets_at"] = composer.time.monotonic() + 30

    def _fail_if_called(*a, **kw):
        raise AssertionError("httpx.post should not be called when budget is exhausted")

    monkeypatch.setattr(composer.httpx, "post", _fail_if_called)
    assert composer._call_llm("system", "user") is None


def test_provider_config_strips_whitespace_from_api_key(monkeypatch):
    # Found live: a trailing space pasted into a hosting provider's env var
    # UI (or left in a local .env) makes httpx raise LocalProtocolError on
    # the Authorization header — invisible in the fallback path since
    # _call_llm swallows all exceptions, so it silently degrades to the
    # template with no error surfaced. Stripping defensively here fixes the
    # whole class of bug regardless of where the stray whitespace came from.
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_test_key \n")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    provider, api_key, model = composer._provider_config()
    assert api_key == "gsk_test_key"


def test_call_llm_sends_low_reasoning_effort_for_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    captured = {}

    class FakeResp:
        headers = {"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-reset-tokens": "5s"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(composer.httpx, "post", _fake_post)
    result = composer._call_llm("system", "user")
    assert result == "ok"
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["max_tokens"] == composer._GROQ_MAX_TOKENS
    assert composer._rate_limit_state["remaining_tokens"] == 5000.0
