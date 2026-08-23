from datetime import datetime, timezone

from bot.selection import is_eligible, rank, score

NOW = datetime(2026, 4, 26, 10, 35, tzinfo=timezone.utc)


def _trigger(**overrides):
    base = {
        "id": "trg_x", "scope": "merchant", "kind": "perf_dip", "source": "internal",
        "merchant_id": "m_001", "customer_id": None,
        "payload": {}, "urgency": 3, "suppression_key": "sk", "expires_at": "2026-05-10T00:00:00Z",
    }
    base.update(overrides)
    return base


def _merchant(**overrides):
    base = {"merchant_id": "m_001", "identity": {"name": "Test Merchant"}, "signals": []}
    base.update(overrides)
    return base


def _customer(**overrides):
    base = {"customer_id": "c_001", "merchant_id": "m_001", "identity": {"name": "Test Customer"},
            "state": "active", "consent": {"scope": []}}
    base.update(overrides)
    return base


def test_expired_trigger_is_ineligible():
    trigger = _trigger(expires_at="2026-04-20T00:00:00Z")  # before NOW
    assert not is_eligible(trigger, NOW, _merchant())


def test_unknown_merchant_is_ineligible():
    assert not is_eligible(_trigger(), NOW, None)


def test_customer_scope_without_customer_is_ineligible():
    trigger = _trigger(scope="customer", kind="recall_due", customer_id="c_001")
    assert not is_eligible(trigger, NOW, _merchant(), None)


def test_consent_gating_mapped_kind_blocks_wrong_scope():
    trigger = _trigger(scope="customer", kind="recall_due", customer_id="c_001")
    customer = _customer(consent={"scope": ["promotional_offers"]})  # no recall_reminders
    assert not is_eligible(trigger, NOW, _merchant(), customer)


def test_consent_gating_mapped_kind_allows_matching_scope():
    trigger = _trigger(scope="customer", kind="recall_due", customer_id="c_001")
    customer = _customer(consent={"scope": ["recall_reminders"]})
    assert is_eligible(trigger, NOW, _merchant(), customer)


def test_consent_gating_unmapped_kind_needs_any_nonempty_consent():
    trigger = _trigger(scope="customer", kind="some_future_kind", customer_id="c_001")
    assert not is_eligible(trigger, NOW, _merchant(), _customer(consent={"scope": []}))
    assert is_eligible(trigger, NOW, _merchant(), _customer(consent={"scope": ["anything"]}))


def test_score_increases_with_urgency():
    low = score(_trigger(urgency=1), NOW, _merchant())
    high = score(_trigger(urgency=5), NOW, _merchant())
    assert high > low


def test_score_increases_as_expiry_approaches():
    soon = score(_trigger(expires_at="2026-04-26T12:00:00Z"), NOW, _merchant())
    later = score(_trigger(expires_at="2026-06-01T00:00:00Z"), NOW, _merchant())
    assert soon > later


def test_score_boosted_by_matching_signal_token():
    trigger = _trigger(kind="dormant_with_vera")
    no_signal = score(trigger, NOW, _merchant(signals=["ctr_below_peer_median"]))
    with_signal = score(trigger, NOW, _merchant(signals=["dormant"]))
    assert with_signal > no_signal


def test_rank_filters_and_orders_deterministically():
    triggers = [
        _trigger(id="trg_b", urgency=2, expires_at="2026-06-01T00:00:00Z"),
        _trigger(id="trg_a", urgency=5, expires_at="2026-04-27T00:00:00Z"),
        _trigger(id="trg_expired", expires_at="2026-01-01T00:00:00Z"),
    ]
    ranked = rank(triggers, NOW, {"m_001": _merchant()}, {})
    assert [t["id"] for t in ranked] == ["trg_a", "trg_b"]


def test_rank_tiebreak_is_deterministic_trigger_id_order():
    triggers = [_trigger(id="trg_z"), _trigger(id="trg_a")]  # identical scores
    ranked = rank(triggers, NOW, {"m_001": _merchant()}, {})
    assert [t["id"] for t in ranked] == ["trg_a", "trg_z"]
