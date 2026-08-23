"""Deterministic (merchant, trigger) scoring and eligibility — no LLM involved.

CLAUDE.md is explicit that this module stays pure Python: decision quality
(which trigger to act on, and in what order) is graded/reasoned about
separately from message quality (composer.py), so it needs to be
independently testable and fully reproducible — same inputs, same ranking,
every time. It operates on raw context dicts, the same shape store.py hands
back — no pydantic parsing needed here.

Two stages:

  1. is_eligible() — a hard yes/no gate, checked before any scoring happens.
     A trigger is eligible only if:
       - its merchant is actually known to us
       - it hasn't expired relative to the tick's `now`
       - for customer-scope triggers, the customer is known AND has actually
         consented to this *kind* of outreach (challenge-brief.md §4.4 /
         testing-brief §3.3's consent.scope field exists precisely so a bot
         doesn't message a customer about something they never opted into).
         REQUIRED_CONSENT maps trigger kind -> acceptable consent.scope
         values, taken from the brief's own examples (Priya's consent scope
         literally lists "recall_reminders"/"appointment_reminders" matching
         her trigger kinds). A kind outside that table just needs *some*
         non-empty consent — a conservative default, since guessing a
         mapping we have no evidence for is worse than requiring proof of
         opt-in. Logged in problemFaced.txt.
     This is intentionally separate from suppression.py: consent asks "is
     this customer-scope send legal at all," suppression asks "have we
     already sent this exact suppression_key before." Different questions,
     different modules.

  2. score() — for everything that passes the gate, a single deterministic
     number: urgency * recency * signal_match.
       - urgency: trigger.urgency (1-5), straight from the data.
       - recency: time-pressure from data the judge actually controls — how
         soon trigger.expires_at is relative to the tick's own `now`. NOT
         "when did our server receive this" — that would violate
         determinism (store.py's stored_at is a real wall-clock read,
         explicitly documented there as never feeding into decisions,
         because two identical judge runs could push the same trigger at
         different real-world instants due to nothing but network jitter).
       - signal_match: a light, generic tie-breaker — token overlap between
         trigger.kind and merchant.signals (e.g. "dormant_with_vera" vs a
         "dormant" signal), or between trigger.kind and customer.state for
         customer-scope triggers (e.g. "customer_lapsed_soft" vs
         state="lapsed_soft"). Deliberately NOT a semantic relevance engine —
         reasoning like Case Study 1's "your high-risk adult patients"
         callout requires cross-referencing the trigger's payload against
         category digest content, which is exactly the kind of judgment
         composer.py's LLM call is for, with full context in front of it.
         Making selection.py hand-author that would mean guessing at domain
         judgment the brief never specifies — so here it's a modest nudge,
         not the primary ranking driver.

rank() ties the two together: filters to eligible candidates, sorts by score
descending, then by trigger id ascending as a deterministic tiebreak (so two
equal-score candidates always land in the same order, regardless of dict
iteration).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

REQUIRED_CONSENT: dict[str, set[str]] = {
    "recall_due": {"recall_reminders"},
    "chronic_refill_due": {"recall_reminders"},
    "appointment_tomorrow": {"appointment_reminders"},
    "customer_lapsed_soft": {"recall_reminders", "promotional_offers"},
    "customer_lapsed_hard": {"recall_reminders", "promotional_offers"},
    "trial_followup": {"promotional_offers", "appointment_reminders"},
}

_STOPWORD_TOKENS = {"with", "due", "upcoming", "reached", "emerged", "change", "changed", "followup"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[_:\s]+", s.lower()) if t and t not in _STOPWORD_TOKENS and not t.isdigit()}


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_eligible(
    trigger: dict[str, Any],
    now: datetime,
    merchant: Optional[dict[str, Any]],
    customer: Optional[dict[str, Any]] = None,
) -> bool:
    if merchant is None:
        return False

    expires_at = _parse_iso(trigger.get("expires_at"))
    if expires_at is not None and expires_at <= now:
        return False

    if trigger.get("scope") == "customer":
        if customer is None:
            return False
        consent_scope = set(customer.get("consent", {}).get("scope", []))
        required = REQUIRED_CONSENT.get(trigger.get("kind", ""))
        if required is not None:
            if not (consent_scope & required):
                return False
        elif not consent_scope:
            return False

    return True


def score(
    trigger: dict[str, Any],
    now: datetime,
    merchant: Optional[dict[str, Any]] = None,
    customer: Optional[dict[str, Any]] = None,
) -> float:
    urgency = float(trigger.get("urgency", 3))

    expires_at = _parse_iso(trigger.get("expires_at"))
    if expires_at is None:
        recency = 1.0
    else:
        hours_left = max((expires_at - now).total_seconds() / 3600.0, 0.1)
        recency = 1.0 + (24.0 / hours_left)  # <1 day left roughly doubles the score; weeks out barely moves it

    kind_tokens = _tokens(trigger.get("kind", ""))
    signal_tokens: set[str] = set()
    if merchant is not None:
        for s in merchant.get("signals", []):
            signal_tokens |= _tokens(s)
    if customer is not None:
        signal_tokens |= _tokens(customer.get("state", ""))
    overlap = len(kind_tokens & signal_tokens)
    signal_match = 1.0 + 0.5 * overlap

    return urgency * recency * signal_match


def rank(
    triggers: list[dict[str, Any]],
    now: datetime,
    merchants_by_id: dict[str, dict[str, Any]],
    customers_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filters to eligible triggers and returns them sorted best-first."""
    scored = []
    for trg in triggers:
        merchant = merchants_by_id.get(trg.get("merchant_id", ""))
        customer_id = trg.get("customer_id")
        customer = customers_by_id.get(customer_id) if customer_id else None
        if not is_eligible(trg, now, merchant, customer):
            continue
        scored.append((score(trg, now, merchant, customer), trg))

    scored.sort(key=lambda pair: (-pair[0], pair[1].get("id", "")))
    return [trg for _, trg in scored]
