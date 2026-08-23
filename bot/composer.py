"""The LLM call (temp=0, strict JSON out) plus a deterministic fallback template.

compose() matches challenge-brief.md §5's exact signature: (category,
merchant, trigger, customer=None) -> ComposedMessage. All four are raw dicts
— the same shape store.py hands back — consistent with how selection.py and
voice.py already operate, so nothing upstream needs to parse into pydantic
models before calling this.

Design choices worth calling out:

  - send_as and suppression_key are NEVER left to the LLM. send_as is
    mechanically "merchant_on_behalf" iff a customer is passed, else "vera";
    suppression_key is always trigger["suppression_key"] verbatim. Letting
    the LLM produce these risked typos/drift for zero benefit — they're
    fully determined by the inputs already, so this also shrinks the LLM's
    job down to the part that actually needs judgment (body, cta, claims).

  - ONE prompt template, parameterized by CategoryContext.voice, rather than
    N hardcoded per-category or per-trigger-kind template strings.
    engagement-research.md (background, not authoritative) suggests
    per-kind prompt variants; I'm not building that — voice differences are
    already data (vocab_allowed/vocab_taboo/tone), so the same prompt
    skeleton naturally adapts per category without a template-per-category
    file. Building N near-duplicate templates now would be designing for a
    hypothetical need the actual dataset doesn't demonstrate yet. Logged in
    problemFaced.txt.

  - Determinism is enforced by memoizing on a hash of the full input bundle,
    not by trusting the LLM provider's temp=0 to be bit-identical across
    calls (it often isn't, especially under batched inference). Identical
    (category, merchant, trigger, customer) always returns the exact same
    ComposedMessage for the lifetime of the process, regardless of what the
    live LLM call would return on a second try. This is the actual fix for
    CLAUDE.md's "same inputs -> same output" hard requirement — logged.

  - No re-prompt/self-repair loop on a bad first LLM response. The brief's
    §13 suggests "re-prompt if it fails" as one option; I chose not to
    implement it for v1 — another network round-trip risks the judge's 30s
    budget and still isn't guaranteed to succeed, whereas the deterministic
    fallback is already required regardless and is instant. Logged.

  - The LLM call itself is capped well under the judge's 30s per-call budget
    (testing-brief §5) so there's always time left to fall back and still
    respond before the judge's own timeout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from bot.grounding import Claim, check_grounding
from bot.models import ComposedMessage
from bot.voice import count_ctas, find_taboo_violations, load_voice

load_dotenv()

LLM_TIMEOUT_SECONDS = 12.0

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "deepseek": "deepseek-chat",
}

_OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
}

# Conventional provider-specific names, checked as a fallback so "however
# the user naturally named their key" just works without them needing to
# know this module's own generic LLM_API_KEY convention.
_PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

_CACHE: dict[str, ComposedMessage] = {}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]] = None,
) -> ComposedMessage:
    cache_key = _cache_key(category, merchant, trigger, customer)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    send_as = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get("suppression_key", "")
    voice = load_voice(category)
    bundle = {"category": category, "merchant": merchant, "trigger": trigger, "customer": customer}

    result = _try_llm_compose(merchant, trigger, customer, voice, bundle, send_as, suppression_key)
    if result is None:
        body, cta = _compose_fallback(merchant, trigger, customer)
        result = ComposedMessage(
            body=body, cta=cta, send_as=send_as, suppression_key=suppression_key,
            rationale="Deterministic fallback template — LLM unavailable, timed out, or its output failed validation.",
        )

    _CACHE[cache_key] = result
    return result


def _cache_key(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    blob = json.dumps(
        {"category": category, "merchant": merchant, "trigger": trigger, "customer": customer},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

def _provider_config() -> tuple[str, str, str]:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    provider_specific = os.environ.get(_PROVIDER_ENV_KEYS.get(provider, ""), "")
    api_key = os.environ.get("LLM_API_KEY") or provider_specific
    model = os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(provider, "")
    return provider, api_key, model


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    provider, api_key, model = _provider_config()
    if not api_key:
        return None
    try:
        if provider == "anthropic":
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": model, "max_tokens": 2000, "temperature": 0,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                headers={
                    "x-api-key": api_key, "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

        url = _OPENAI_COMPATIBLE_URLS.get(provider)
        if url is None:
            return None
        resp = httpx.post(
            url,
            json={
                "model": model, "temperature": 0, "max_tokens": 2000,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            timeout=LLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _build_system_prompt(voice) -> str:
    return f"""You are Vera, magicpin's WhatsApp AI assistant for merchant growth. You write ONE outbound \
message given category, merchant, trigger, and (optionally) customer context.

VOICE FOR THIS CATEGORY:
- tone: {voice.tone or "neutral"}; register: {voice.register_ or "professional"}
- welcome vocabulary: {", ".join(voice.vocab_allowed) or "none specified"}
- NEVER use these words/phrases, or any close equivalent: {", ".join(voice.vocab_taboo) or "none specified"}
- salutation style examples: {", ".join(voice.salutation_examples) or "none specified"}

HARD RULES:
1. Every verifiable fact you state (a number, date, statistic, price, name) MUST come from the context \
provided below. Never invent one, even a plausible-sounding one.
2. For every such fact, add an entry to "claims": {{"text": <the fact as you phrased it>, "source_path": \
<dotted path into the bundle, e.g. "category.digest[0].trial_n" or "merchant.performance.views" or \
"trigger.payload.due_date">}}. Use [N] for list indices.
3. One primary call-to-action, either a single question at the very end of the message, or a direct \
imperative instruction (e.g. "Reply 1 for Wed, 2 for Thu" for a multi_choice_slot cta) — never both, and \
never more than one distinct ask. Use cta="none" only for purely informational messages with no ask at all.
4. Never include a URL.
5. Never re-introduce yourself if merchant.conversation_history is non-empty.
6. Match the merchant's or customer's language preference — Hindi-English code-mix is fine and often \
preferred when languages include "hi" or language_pref mentions Hindi.
7. No long preambles ("I hope you're doing well..."). Get to the point in the first sentence.
8. Keep it concise — 2-4 sentences, matching the length of real good examples.
9. If a customer is provided, address the customer directly (this message goes out from the merchant's \
own WhatsApp number, on their behalf) — do not talk to the merchant instead.
10. When citing a source, use the human-readable "source" field text (e.g. "JIDA Oct 2026, p.14") — NEVER \
an internal "id" field (e.g. "d_2026W17_jida_fluoride"). Internal ids are only for your claims' \
source_path, never for the message body itself.

Respond with ONLY a single JSON object — no markdown fences, no commentary before or after — matching \
exactly this shape:
{{"body": "...", "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | \
"none", "rationale": "...", "claims": [{{"text": "...", "source_path": "..."}}]}}"""


def _build_user_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], send_as: str) -> str:
    audience = (
        f"CUSTOMER-FACING (send_as={send_as}) — this message goes to the merchant's own customer, not to the merchant."
        if customer is not None
        else f"MERCHANT-FACING (send_as={send_as}) — this message goes to the merchant themself."
    )
    return f"""{audience}

category = {json.dumps(category, ensure_ascii=False)}

merchant = {json.dumps(merchant, ensure_ascii=False)}

trigger = {json.dumps(trigger, ensure_ascii=False)}

customer = {json.dumps(customer, ensure_ascii=False) if customer is not None else "null"}

Write the message now."""


def _parse_llm_json(raw: str) -> Optional[dict]:
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _passes_validation(body: str, cta: str, claims: list[Claim], voice, bundle: dict[str, Any]) -> bool:
    if _URL_RE.search(body):
        return False
    if find_taboo_violations(body, voice):
        return False

    # NOT requiring exactly one '?' when cta != "none", and NOT gating on
    # has_buried_cta — both were disproven by the real case studies during
    # self-critique. Case Study 2's gold CTA ("Reply 1 for Wed, 2 for Thu...")
    # has zero question marks; Case Study 4's gold CTA is a question in the
    # MIDDLE of the message with two more legitimate sentences after it.
    # Only a hard ceiling remains: 2+ questions is still a real multi-CTA
    # spam signal (that pattern doesn't appear in any gold example).
    n_ctas = count_ctas(body)
    if cta == "none":
        if n_ctas != 0:
            return False
    else:
        if n_ctas > 1:
            return False

    return check_grounding(body, claims, bundle).ok


def _try_llm_compose(
    merchant: dict, trigger: dict, customer: Optional[dict], voice, bundle: dict[str, Any],
    send_as: str, suppression_key: str,
) -> Optional[ComposedMessage]:
    system_prompt = _build_system_prompt(voice)
    user_prompt = _build_user_prompt(bundle["category"], merchant, trigger, customer, send_as)

    raw = _call_llm(system_prompt, user_prompt)
    if raw is None:
        return None

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return None

    body = str(parsed.get("body", "")).strip()
    cta = str(parsed.get("cta", "none"))
    rationale = str(parsed.get("rationale", "")).strip()
    if not body or not rationale:
        return None

    try:
        claims = [Claim(**c) for c in parsed.get("claims", [])]
    except Exception:
        return None

    if not _passes_validation(body, cta, claims, voice, bundle):
        return None

    return ComposedMessage(body=body, cta=cta, send_as=send_as, suppression_key=suppression_key, rationale=rationale)


# ---------------------------------------------------------------------------
# Deterministic fallback — pure data substitution, zero invented facts
# ---------------------------------------------------------------------------

def _compose_fallback(merchant: dict, trigger: dict, customer: Optional[dict]) -> tuple[str, str]:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload") or {}
    merchant_name = merchant.get("identity", {}).get("name", "your business")
    owner = merchant.get("identity", {}).get("owner_first_name")

    if customer is not None:
        cust_name = customer.get("identity", {}).get("name") or "there"
        greeting = f"Hi {cust_name}, {merchant_name} here"
    else:
        greeting = f"Hi {owner}" if owner else f"Hi, {merchant_name} team"

    if kind == "perf_dip" and "metric" in payload and "delta_pct" in payload:
        pct = abs(float(payload["delta_pct"])) * 100
        window = payload.get("window", "week")
        body = f"{greeting} — your {payload['metric']} dropped {pct:.0f}% this {window}. Want me to take a look with you?"
        return body, "binary_yes_no"

    if kind == "perf_spike" and "metric" in payload:
        body = f"{greeting} — your {payload['metric']} is trending up. Want me to help you keep the momentum going?"
        return body, "binary_yes_no"

    if kind == "renewal_due" and "days_remaining" in payload:
        body = f"{greeting} — your subscription renews in {payload['days_remaining']} days. Want me to walk you through the options?"
        return body, "binary_yes_no"

    if kind in ("recall_due", "chronic_refill_due", "appointment_tomorrow") and payload.get("due_date"):
        body = f"{greeting} — a visit is due around {payload['due_date']}. Want us to hold a slot for you?"
        return body, "binary_yes_no"

    if kind == "research_digest":
        body = f"{greeting} — there's a new item in this week's category digest that may be relevant to you. Want me to send it over?"
        return body, "open_ended"

    body = f"{greeting} — I noticed something worth flagging for your business. Want me to share the details?"
    return body, "open_ended"
