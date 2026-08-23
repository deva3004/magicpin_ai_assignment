"""Pydantic schemas for the Vera compose engine.

Two families of model live here:
  1. The 4 context schemas (Category/Merchant/Trigger/Customer) that mirror
     the dataclasses in challenge-brief.md §4 — these represent dataset /
     judge-pushed content.
  2. The request/response envelopes for the 5 HTTP endpoints defined in
     challenge-testing-brief.md §2 — these represent our wire contract.

Context schemas use `extra="allow"` throughout: the dataset carries fields
beyond what the briefs document (review_themes, established_year, ctr_pct,
...), and the judge can push post-submission context with fields we've never
seen. Rejecting on unknown fields would violate the testing-brief's own FAQ
("the bot should be ready for any context to arrive at any time").

Fields that describe domain data we only *receive* (trigger.kind,
customer.state, subscription.status, ...) are typed as plain str rather than
Literal enums, for the same reason — an unfamiliar value should be carried
through, not rejected. Literal is reserved for values we control ourselves
(our own send_as, our own reply action).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LenientModel(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# CategoryContext
# ---------------------------------------------------------------------------

class VoiceProfile(LenientModel):
    tone: Optional[str] = None
    register: Optional[str] = None
    code_mix: Optional[str] = None
    vocab_allowed: list[str] = Field(default_factory=list)
    vocab_taboo: list[str] = Field(default_factory=list)
    salutation_examples: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)


class OfferTemplate(LenientModel):
    id: Optional[str] = None
    title: str
    value: Optional[str] = None
    audience: Optional[str] = None
    type: Optional[str] = None


class PeerStats(LenientModel):
    scope: Optional[str] = None
    avg_rating: Optional[float] = None
    avg_review_count: Optional[int] = None
    avg_ctr: Optional[float] = None


class DigestItem(LenientModel):
    id: str
    kind: Optional[str] = None
    title: str
    source: Optional[str] = None
    summary: Optional[str] = None
    actionable: Optional[str] = None


class ContentItem(LenientModel):
    id: str
    title: str
    channel: Optional[str] = None
    body: Optional[str] = None


class SeasonalBeat(LenientModel):
    month_range: str
    note: str


class TrendSignal(LenientModel):
    query: str
    delta_yoy: Optional[float] = None
    segment_age: Optional[str] = None
    skew: Optional[str] = None


class CategoryContext(LenientModel):
    slug: str
    display_name: Optional[str] = None
    voice: VoiceProfile = Field(default_factory=VoiceProfile)
    offer_catalog: list[OfferTemplate] = Field(default_factory=list)
    peer_stats: PeerStats = Field(default_factory=PeerStats)
    digest: list[DigestItem] = Field(default_factory=list)
    patient_content_library: list[ContentItem] = Field(default_factory=list)
    seasonal_beats: list[SeasonalBeat] = Field(default_factory=list)
    trend_signals: list[TrendSignal] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MerchantContext
# ---------------------------------------------------------------------------

class Identity(LenientModel):
    name: str
    city: Optional[str] = None
    locality: Optional[str] = None
    place_id: Optional[str] = None
    verified: Optional[bool] = None
    languages: list[str] = Field(default_factory=list)
    owner_first_name: Optional[str] = None


class Subscription(LenientModel):
    status: str = "unknown"
    plan: Optional[str] = None
    days_remaining: Optional[int] = None


class PerformanceSnapshot(LenientModel):
    window_days: Optional[int] = None
    views: Optional[int] = None
    calls: Optional[int] = None
    directions: Optional[int] = None
    ctr: Optional[float] = None
    leads: Optional[int] = None
    delta_7d: dict[str, Any] = Field(default_factory=dict)


class MerchantOffer(LenientModel):
    id: Optional[str] = None
    title: str
    status: str


class ConversationTurn(LenientModel):
    # "from" is a Python keyword, hence the alias — this is the one field in
    # the whole schema set that can't just be typed straight off the JSON key.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    ts: Optional[str] = None
    from_: str = Field(alias="from")
    body: str
    engagement: Optional[str] = None


class CustomerAggregate(LenientModel):
    total_unique_ytd: Optional[int] = None
    lapsed_180d_plus: Optional[int] = None
    retention_6mo_pct: Optional[float] = None
    high_risk_adult_count: Optional[int] = None


class MerchantContext(LenientModel):
    merchant_id: str
    category_slug: str
    identity: Identity
    subscription: Subscription = Field(default_factory=Subscription)
    performance: PerformanceSnapshot = Field(default_factory=PerformanceSnapshot)
    offers: list[MerchantOffer] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    customer_aggregate: CustomerAggregate = Field(default_factory=CustomerAggregate)
    signals: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TriggerContext
# ---------------------------------------------------------------------------

class TriggerContext(LenientModel):
    id: str
    scope: str  # "merchant" | "customer" in every example seen, not enforced
    kind: str
    source: str  # "external" | "internal"
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int = 3
    suppression_key: str
    expires_at: Optional[str] = None


# ---------------------------------------------------------------------------
# CustomerContext
# ---------------------------------------------------------------------------

class CustomerIdentity(LenientModel):
    name: str
    phone_redacted: Optional[str] = None
    language_pref: Optional[str] = None


class Relationship(LenientModel):
    first_visit: Optional[str] = None
    last_visit: Optional[str] = None
    visits_total: Optional[int] = None
    services_received: list[str] = Field(default_factory=list)
    lifetime_value: Optional[float] = None


class Preferences(LenientModel):
    channel: Optional[str] = None
    preferred_slots: Optional[str] = None
    reminder_opt_in: Optional[bool] = None


class Consent(LenientModel):
    opted_in_at: Optional[str] = None
    scope: list[str] = Field(default_factory=list)


class CustomerContext(LenientModel):
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: Relationship = Field(default_factory=Relationship)
    state: str = "active"
    preferences: Preferences = Field(default_factory=Preferences)
    consent: Consent = Field(default_factory=Consent)


# ---------------------------------------------------------------------------
# Internal composer output (challenge-brief.md §5)
# ---------------------------------------------------------------------------

class ComposedMessage(BaseModel):
    body: str
    cta: str
    send_as: Literal["vera", "merchant_on_behalf"]
    suppression_key: str
    rationale: str


# ---------------------------------------------------------------------------
# POST /v1/context — challenge-testing-brief.md §2.1
# ---------------------------------------------------------------------------

class ContextPushRequest(BaseModel):
    # scope is intentionally str, not Literal: an invalid value must produce
    # the spec's exact 400 body via app.py, not FastAPI's generic 422.
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


class ContextPushAccepted(BaseModel):
    accepted: Literal[True] = True
    ack_id: str
    stored_at: str


class ContextPushRejected(BaseModel):
    accepted: Literal[False] = False
    reason: str
    current_version: Optional[int] = None
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# POST /v1/tick — challenge-testing-brief.md §2.2
# ---------------------------------------------------------------------------

class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: Literal["vera", "merchant_on_behalf"]
    trigger_id: str
    template_name: str
    template_params: list[str] = Field(default_factory=list)
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: list[TickAction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /v1/reply — challenge-testing-brief.md §2.3
# ---------------------------------------------------------------------------

class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


class ReplyResponse(BaseModel):
    action: Literal["send", "wait", "end"]
    body: Optional[str] = None
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str


# ---------------------------------------------------------------------------
# GET /v1/healthz, GET /v1/metadata — challenge-testing-brief.md §2.4-2.5
# ---------------------------------------------------------------------------

class HealthzResponse(BaseModel):
    status: Literal["ok"] = "ok"
    uptime_seconds: int
    contexts_loaded: dict[str, int]


class MetadataResponse(BaseModel):
    team_name: str
    team_members: list[str] = Field(default_factory=list)
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str


# ---------------------------------------------------------------------------
# test_pairs.json entries — consumed by scripts/generate_submission.py
# ---------------------------------------------------------------------------

class TestPair(BaseModel):
    test_id: str
    trigger_id: str
    merchant_id: str
    customer_id: Optional[str] = None
