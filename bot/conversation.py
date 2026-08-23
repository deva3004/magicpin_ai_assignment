"""The /v1/reply state machine.

Owns per-conversation state and the deterministic ROUTING decision (auto-reply
detection, intent-transition, hostile/opt-out handling, graceful exit) — pure
Python, no LLM, same philosophy as selection.py. Content generation for an
actual "send" action is delegated to composer.compose_reply(), which reuses
composer.py's existing LLM/grounding/voice validation pipeline rather than
duplicating it.

Auto-reply detection reconciles a real contradiction found between the two
authoritative docs (logged in problemFaced.txt):
  - testing-brief.md Example 2.5 shows a FIRST-occurrence canned auto-reply
    getting action="wait" immediately.
  - api-call-examples.md Example 4.1's explicitly-labeled "Good bot response"
    shows a FIRST-occurrence canned auto-reply getting action="send" (one
    light nudge), THEN wait on the 2nd occurrence, THEN end on the 3rd — a
    3-strike escalation matching challenge-brief.md §12's own hint ("same
    message verbatim 3+ times = auto-reply").
  Chose the 4.1 escalation (send -> wait -> end): it's the more detailed,
  explicitly-labeled reference, and it's a strictly richer behavior (one real
  attempt to reach a human before backing off) that still satisfies
  testing-brief's general auto-reply-detection requirement. Deviates from the
  single wait-on-first-contact example — a logged judgment call, not a
  silent guess.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from bot.composer import compose_reply

# Common WhatsApp Business canned-reply phrasing — content-based detection,
# fires even on the very first message in a conversation (unlike the
# repeat-count signal below, which by definition can't fire until a message
# has been seen before).
_AUTO_REPLY_PATTERNS = [
    "thank you for contacting", "will respond shortly", "we have received your message",
    "this is an automated", "automated reply", "automated response", "out of office",
    "currently unavailable", "our team will get back to you", "we will get back to you shortly",
]

_OPTOUT_HOSTILE_PATTERNS = [
    "stop messaging", "stop contacting", "unsubscribe", "not interested", "leave me alone",
    "don't message", "do not message", "don't contact", "useless", "spam", "bothering me",
    "waste of time", "stop sending", "stop texting",
]

_COMMITMENT_PATTERNS = [
    "let's do it", "lets do it", "let us do it", "go ahead", "sounds good", "sign me up",
    "count me in", "i want to join", "yes let's", "ok let's", "okay let's", "yes, let's",
    "want to join", "please proceed", "let's go", "lets go",
]

# challenge-brief.md §12.5: "gracefully exit... after 3 unanswered nudges."
# /v1/tick can only ever START a new conversation_id (testing-brief §2.2:
# "Reusing an existing conversation_id is invalid here"), so every
# continuation happens through /v1/reply — meaning WE only ever get to send
# again in response to an incoming reply. This caps total sends per
# conversation so an inconclusive back-and-forth can't run forever.
MAX_SENDS_WITHOUT_RESOLUTION = 3


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _matches_any(text: str, patterns: list[str]) -> bool:
    lowered = _norm(text)
    return any(p in lowered for p in patterns)


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    category_slug: str
    trigger_id: str
    send_as: str
    ended: bool = False
    sent_bodies: list[str] = field(default_factory=list)
    received_texts: list[str] = field(default_factory=list)  # normalized, for auto-reply repeat counting
    send_count: int = 0


class ConversationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ConversationState] = {}

    def start(
        self, conversation_id: str, merchant_id: str, customer_id: Optional[str],
        category_slug: str, trigger_id: str, send_as: str, initial_body: str,
    ) -> ConversationState:
        state = ConversationState(
            conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id,
            category_slug=category_slug, trigger_id=trigger_id, send_as=send_as,
            sent_bodies=[initial_body], send_count=1,
        )
        with self._lock:
            self._states[conversation_id] = state
        return state

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        return self._states.get(conversation_id)

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


def handle_reply(
    state: ConversationState,
    message: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Returns a dict with an "action" key ("send" | "wait" | "end") plus that
    action's other fields — the shape /v1/reply's response needs, minus
    nothing (app.py can pass this straight through)."""

    if state.ended:
        return {"action": "end", "rationale": "Conversation already closed; not sending further messages."}

    if _matches_any(message, _OPTOUT_HOSTILE_PATTERNS):
        state.ended = True
        return {
            "action": "end",
            "rationale": "Explicit opt-out or hostility detected; closing gracefully, no further sends on this conversation.",
        }

    normalized = _norm(message)
    prior_repeats = state.received_texts.count(normalized)
    state.received_texts.append(normalized)

    if prior_repeats >= 2:
        state.ended = True
        return {
            "action": "end",
            "rationale": "Same message received verbatim 3+ times — auto-reply pattern confirmed, no real engagement signal; closing.",
        }
    if prior_repeats == 1:
        return {
            "action": "wait", "wait_seconds": 86400,
            "rationale": "Same message received twice — likely an auto-reply; backing off 24h for the owner to see it.",
        }
    if prior_repeats == 0 and _matches_any(message, _AUTO_REPLY_PATTERNS):
        nudge = "Looks like an auto-reply 😊 When you get a chance, just reply to let me know what you'd like to do next."
        state.sent_bodies.append(nudge)
        state.send_count += 1
        return {
            "action": "send", "body": nudge, "cta": "open_ended",
            "rationale": "Detected canned auto-reply phrasing; one explicit prompt to flag it for the owner before backing off.",
        }

    if state.send_count >= MAX_SENDS_WITHOUT_RESOLUTION:
        state.ended = True
        return {
            "action": "end",
            "rationale": f"Sent {state.send_count} messages in this conversation with no clear commitment or "
                         "decline; gracefully exiting rather than persisting indefinitely.",
        }

    intent_hint = "explicit_commitment" if _matches_any(message, _COMMITMENT_PATTERNS) else None

    reply = compose_reply(
        category=category, merchant=merchant, trigger=trigger, customer=customer,
        prior_bodies=state.sent_bodies, latest_message=message, intent_hint=intent_hint,
    )

    if reply.body in state.sent_bodies:
        return {
            "action": "end",
            "rationale": "Would have repeated a message already sent verbatim in this conversation "
                         "(anti-repetition); closing instead.",
        }

    state.sent_bodies.append(reply.body)
    state.send_count += 1
    return {"action": "send", "body": reply.body, "cta": reply.cta, "rationale": reply.rationale}
