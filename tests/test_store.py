from bot.store import ContextStore, MAX_PAYLOAD_BYTES


def test_first_push_accepted():
    store = ContextStore()
    result = store.push("merchant", "m_001", 1, {"name": "Dr. Meera"})
    assert result.accepted
    assert result.ack_id == "ack_m_001_v1"
    assert store.get("merchant", "m_001") == {"name": "Dr. Meera"}


def test_same_version_repush_is_rejected_as_stale():
    store = ContextStore()
    store.push("merchant", "m_001", 1, {"name": "v1"})
    result = store.push("merchant", "m_001", 1, {"name": "v1 again"})
    assert not result.accepted
    assert result.reason == "stale_version"
    assert result.current_version == 1
    # no-op: original payload untouched
    assert store.get("merchant", "m_001") == {"name": "v1"}


def test_lower_version_rejected():
    store = ContextStore()
    store.push("merchant", "m_001", 5, {"name": "v5"})
    result = store.push("merchant", "m_001", 3, {"name": "v3"})
    assert not result.accepted
    assert result.current_version == 5


def test_higher_version_replaces_atomically():
    store = ContextStore()
    store.push("merchant", "m_001", 1, {"name": "v1"})
    result = store.push("merchant", "m_001", 2, {"name": "v2"})
    assert result.accepted
    assert store.get("merchant", "m_001") == {"name": "v2"}
    assert store.get_version("merchant", "m_001") == 2


def test_invalid_scope_rejected():
    store = ContextStore()
    result = store.push("bogus", "x", 1, {})
    assert not result.accepted
    assert result.reason == "invalid_scope"


def test_oversized_payload_rejected():
    store = ContextStore()
    big = {"blob": "x" * (MAX_PAYLOAD_BYTES + 1)}
    result = store.push("merchant", "m_big", 1, big)
    assert not result.accepted
    assert result.reason == "payload_too_large"


def test_counts_reflects_distinct_context_ids_not_push_count():
    store = ContextStore()
    store.push("merchant", "m_001", 1, {})
    store.push("merchant", "m_001", 2, {})  # version bump, same context_id
    store.push("merchant", "m_002", 1, {})
    counts = store.counts()
    assert counts["merchant"] == 2
    assert counts["category"] == 0


def test_all_sorted_by_context_id_regardless_of_push_order():
    store = ContextStore()
    store.push("merchant", "m_003", 1, {"n": 3})
    store.push("merchant", "m_001", 1, {"n": 1})
    store.push("merchant", "m_002", 1, {"n": 2})
    assert list(store.all("merchant").keys()) == ["m_001", "m_002", "m_003"]


def test_clear_wipes_everything():
    store = ContextStore()
    store.push("merchant", "m_001", 1, {})
    store.clear()
    assert store.counts() == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
