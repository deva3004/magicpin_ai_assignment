"""FastAPI service exposing the 5 endpoints challenge-testing-brief.md §2 requires
(plus the optional /v1/teardown from §11), wiring together every other module.

Design choices worth calling out:

  - /v1/context's scope/version validation happens entirely inside
    store.push() (see store.py), and this module just translates its
    PushResult into the exact documented status code/body. Deliberately
    NOT using a strict Pydantic Literal for `scope` on the request model
    (models.py's ContextPushRequest) — that would make FastAPI itself
    reject a bad scope with its own generic 422 shape, before this code
    ever gets a chance to return the spec's exact
    {"accepted": false, "reason": "invalid_scope"} body.

  - /v1/tick caps to ONE action per (merchant_id, customer_id) pair per
    tick, even when multiple triggers are eligible for the same
    recipient — only the highest-ranked one (per selection.rank()'s
    ordering) sends. Nothing in the spec requires firing every eligible
    trigger every tick, and testing-brief's own FAQ says restraint is
    rewarded, spam is penalized.

  - conversation_id is deterministic: f"conv_{merchant_id}_{trigger_id}",
    not a random UUID. Matches case-studies.md's own "decodable, resumable"
    guidance and keeps the whole system reproducible end to end.

  - Every handler is wrapped so an internal exception degrades to a safe
    response ({"actions": []} for tick, action="end" for reply) instead of
    an unhandled 500 — CLAUDE.md is explicit that this must never happen on
    the happy path, and testing-brief's failure-mode table treats a 500 far
    worse than an empty/conservative response.

  - An unknown conversation_id in /v1/reply (one we never started via
    /v1/tick) isn't documented in the spec. Defaulting to action="end"
    rather than raising an error — safer than fabricating context for a
    conversation we know nothing about.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bot.composer import compose
from bot.conversation import ConversationStore, handle_reply
from bot.models import (
    ContextPushRequest, HealthzResponse, MetadataResponse, ReplyRequest, TickRequest,
)
from bot.selection import rank
from bot.store import ContextStore
from bot.suppression import SuppressionStore

app = FastAPI(title="Vera Compose Engine")

START_TIME = time.time()
context_store = ContextStore()
suppression_store = SuppressionStore()
conversation_store = ConversationStore()

MAX_ACTIONS_PER_TICK = 20  # testing-brief §5


def _parse_now(now_str: str) -> datetime:
    try:
        return datetime.fromisoformat(now_str.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# GET /v1/healthz, GET /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz() -> HealthzResponse:
    return HealthzResponse(uptime_seconds=int(time.time() - START_TIME), contexts_loaded=context_store.counts())


@app.get("/v1/metadata")
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        team_name="Devashish Tripathi",
        team_members=["Devashish Tripathi"],
        model="groq/openai-gpt-oss-120b (fallback: deterministic template)",
        approach="4-context composer (category/merchant/trigger/customer) with a pure-Python "
                 "deterministic selection/consent/suppression layer, an LLM call at temperature=0 "
                 "gated by a fact-ledger grounding check + voice/taboo/CTA validation, and a "
                 "deterministic template fallback on any timeout/error/validation failure.",
        contact_email="tripathidevashish07@gmail.com",
        version="0.1.0",
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# POST /v1/context
# ---------------------------------------------------------------------------

@app.post("/v1/context")
async def push_context(body: ContextPushRequest) -> JSONResponse:
    result = context_store.push(body.scope, body.context_id, body.version, body.payload)

    if result.accepted:
        return JSONResponse(
            status_code=200,
            content={"accepted": True, "ack_id": result.ack_id, "stored_at": result.stored_at},
        )
    if result.reason == "stale_version":
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": result.current_version},
        )
    return JSONResponse(
        status_code=400,
        content={"accepted": False, "reason": result.reason, "details": result.details},
    )


# ---------------------------------------------------------------------------
# POST /v1/tick
# ---------------------------------------------------------------------------

@app.post("/v1/tick")
async def tick(body: TickRequest) -> dict[str, Any]:
    try:
        return _tick_impl(body)
    except Exception:
        return {"actions": []}


def _tick_impl(body: TickRequest) -> dict[str, Any]:
    now = _parse_now(body.now)

    triggers = [t for tid in body.available_triggers if (t := context_store.get("trigger", tid)) is not None]
    merchants_by_id = context_store.all("merchant")
    customers_by_id = context_store.all("customer")

    ranked = rank(triggers, now, merchants_by_id, customers_by_id)

    actions: list[dict[str, Any]] = []
    claimed_recipients: set[tuple[str, str]] = set()  # (merchant_id, customer_id or "")

    for trigger in ranked:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break

        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")
        recipient_key = (merchant_id, customer_id or "")
        if recipient_key in claimed_recipients:
            continue

        suppression_key = trigger.get("suppression_key", "")
        if suppression_store.has_fired(merchant_id, suppression_key, customer_id):
            continue

        merchant = merchants_by_id.get(merchant_id)
        if merchant is None:
            continue
        category = context_store.get("category", merchant.get("category_slug", ""))
        if category is None:
            continue
        customer = customers_by_id.get(customer_id) if customer_id else None

        try:
            composed = compose(category, merchant, trigger, customer)
        except Exception:
            continue

        conversation_id = f"conv_{merchant_id}_{trigger['id']}"
        recipient_name = (
            (customer or {}).get("identity", {}).get("name")
            or merchant.get("identity", {}).get("owner_first_name")
            or merchant.get("identity", {}).get("name", "")
        )

        conversation_store.start(
            conversation_id=conversation_id, merchant_id=merchant_id, customer_id=customer_id,
            category_slug=merchant.get("category_slug", ""), trigger_id=trigger["id"],
            send_as=composed.send_as, initial_body=composed.body,
        )
        suppression_store.mark_fired(merchant_id, suppression_key, customer_id)
        claimed_recipients.add(recipient_key)

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.send_as,
            "trigger_id": trigger["id"],
            "template_name": f"{composed.send_as}_{trigger.get('kind', 'generic')}_v1",
            "template_params": [recipient_name, trigger.get("kind", "")],
            "body": composed.body,
            "cta": composed.cta,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        })

    return {"actions": actions}


# ---------------------------------------------------------------------------
# POST /v1/reply
# ---------------------------------------------------------------------------

@app.post("/v1/reply")
async def reply(body: ReplyRequest) -> dict[str, Any]:
    try:
        return _reply_impl(body)
    except Exception:
        return {"action": "end", "rationale": "Internal error handling this reply; ending gracefully rather than risking a bad send."}


def _reply_impl(body: ReplyRequest) -> dict[str, Any]:
    state = conversation_store.get(body.conversation_id)
    if state is None:
        return {"action": "end", "rationale": "Unknown conversation_id — never started via /v1/tick; nothing to continue."}

    merchant = context_store.get("merchant", state.merchant_id)
    category = context_store.get("category", state.category_slug)
    trigger = context_store.get("trigger", state.trigger_id)
    customer = context_store.get("customer", state.customer_id) if state.customer_id else None

    if merchant is None or category is None or trigger is None:
        return {"action": "end", "rationale": "Context for this conversation is no longer available; ending gracefully."}

    return handle_reply(state, body.message, category, merchant, trigger, customer)


# ---------------------------------------------------------------------------
# POST /v1/teardown — optional, testing-brief §11
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown() -> dict[str, bool]:
    context_store.clear()
    suppression_store.clear()
    conversation_store.clear()
    return {"ok": True}
