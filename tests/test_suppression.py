from bot.suppression import SuppressionStore


def test_unfired_key_reports_false():
    store = SuppressionStore()
    assert not store.has_fired("m_001", "research:dentists:2026-W17")


def test_firing_marks_it_for_that_merchant():
    store = SuppressionStore()
    store.mark_fired("m_001", "research:dentists:2026-W17")
    assert store.has_fired("m_001", "research:dentists:2026-W17")


def test_scoping_is_per_recipient_even_when_key_string_is_shared():
    # "research:dentists:2026-W17" carries no merchant id of its own — a naive
    # bare-string dedup would incorrectly suppress it for every dentist.
    store = SuppressionStore()
    store.mark_fired("m_001", "research:dentists:2026-W17")
    assert not store.has_fired("m_099", "research:dentists:2026-W17")


def test_customer_scoped_key_is_scoped_by_customer_not_merchant():
    store = SuppressionStore()
    key = "recall:c_001_priya_for_m001:6mo"
    store.mark_fired("m_001", key, customer_id="c_001_priya_for_m001")
    assert store.has_fired("m_001", key, customer_id="c_001_priya_for_m001")
    assert not store.has_fired("m_001", key, customer_id="c_002_rohit_for_m001")


def test_clear_wipes_all_fired_keys():
    store = SuppressionStore()
    store.mark_fired("m_001", "k")
    store.clear()
    assert not store.has_fired("m_001", "k")
