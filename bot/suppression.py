"""Fired-suppression-key tracker, semantically scoped by recipient.

TriggerContext.suppression_key exists so a given event/content doesn't get
re-sent once it's already fired (challenge-brief.md §4.3, testing-brief
§2.2's action shape). The raw key string is inconsistent about whether it
already embeds a merchant/customer id — compare
"recall:c_001_priya_for_m001:6mo" (customer baked in) against
"research:dentists:2026-W17" (no recipient at all — shared across every
dentist who could receive that week's digest).

If suppression were tracked on the bare key string, sending the research
digest to one dentist would silently suppress it for every other dentist
sharing that key. So this store always compounds the key with the actual
recipient: (customer_id if this is a customer-facing send, else
merchant_id, suppression_key). That's the "semantic scoping" CLAUDE.md
calls for — WHAT (suppression_key) and WHO (recipient) are always tracked
together, regardless of what the raw key string happens to already encode.
"""

from __future__ import annotations

import threading
from typing import Optional


class SuppressionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fired: set[tuple[str, str]] = set()

    @staticmethod
    def _recipient(merchant_id: str, customer_id: Optional[str]) -> str:
        return customer_id if customer_id else merchant_id

    def has_fired(self, merchant_id: str, suppression_key: str, customer_id: Optional[str] = None) -> bool:
        return (self._recipient(merchant_id, customer_id), suppression_key) in self._fired

    def mark_fired(self, merchant_id: str, suppression_key: str, customer_id: Optional[str] = None) -> None:
        key = (self._recipient(merchant_id, customer_id), suppression_key)
        with self._lock:
            self._fired.add(key)

    def clear(self) -> None:
        with self._lock:
            self._fired.clear()
