"""In-memory versioned context store.

Backing store for everything the judge pushes via POST /v1/context — category,
merchant, customer, and trigger payloads. Two hard requirements from
challenge-testing-brief.md §2.1 and §5:

  - Idempotent by (scope, context_id, version): a version <= what's already
    stored is a no-op that reports the current version back (409 territory);
    a strictly higher version replaces the prior one.
  - 500KB payload cap.

Everything lives in one process-local dict guarded by one lock. Concurrency
here is check-then-write (compare incoming version against stored version,
then decide to replace), which needs to be atomic across both steps — a bare
dict assignment alone isn't enough once two pushes for the same key can
interleave. The judge caps itself at 10 req/sec (testing-brief §5), so one
coarse lock costs nothing measurable and avoids per-key lock bookkeeping.

stored_at is a real wall-clock read — that's deliberate and does not leak
into anything compose() decides. It's a receipt timestamp on the ack,
nothing more; the judge's own `now` (from /v1/tick) remains the only clock
composition logic is allowed to reason about.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

MAX_PAYLOAD_BYTES = 500_000  # testing-brief §5: "/v1/context payload size cap | 500 KB"

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@dataclass(frozen=True)
class StoreEntry:
    version: int
    payload: dict[str, Any]
    stored_at: str


@dataclass(frozen=True)
class PushResult:
    accepted: bool
    reason: Optional[str] = None           # set when accepted is False
    current_version: Optional[int] = None  # set when reason == "stale_version"
    details: Optional[str] = None          # set when reason in {"invalid_scope", "payload_too_large"}
    ack_id: Optional[str] = None           # set when accepted is True
    stored_at: Optional[str] = None        # set when accepted is True


class ContextStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], StoreEntry] = {}

    def push(self, scope: str, context_id: str, version: int, payload: dict[str, Any]) -> PushResult:
        if scope not in VALID_SCOPES:
            return PushResult(
                accepted=False, reason="invalid_scope",
                details=f"scope must be one of {sorted(VALID_SCOPES)}, got {scope!r}",
            )

        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size > MAX_PAYLOAD_BYTES:
            return PushResult(
                accepted=False, reason="payload_too_large",
                details=f"payload is {size} bytes, cap is {MAX_PAYLOAD_BYTES}",
            )

        key = (scope, context_id)
        with self._lock:
            current = self._entries.get(key)
            if current is not None and version <= current.version:
                return PushResult(accepted=False, reason="stale_version", current_version=current.version)

            stored_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            self._entries[key] = StoreEntry(version=version, payload=payload, stored_at=stored_at)

        return PushResult(accepted=True, ack_id=f"ack_{context_id}_v{version}", stored_at=stored_at)

    def get(self, scope: str, context_id: str) -> Optional[dict[str, Any]]:
        entry = self._entries.get((scope, context_id))
        return entry.payload if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        entry = self._entries.get((scope, context_id))
        return entry.version if entry else None

    def all(self, scope: str) -> dict[str, dict[str, Any]]:
        """All payloads for a scope, sorted by context_id — deterministic regardless of push order."""
        items = ((cid, e.payload) for (s, cid), e in self._entries.items() if s == scope)
        return dict(sorted(items, key=lambda kv: kv[0]))

    def counts(self) -> dict[str, int]:
        counts = {s: 0 for s in VALID_SCOPES}
        for scope, _cid in self._entries:
            counts[scope] += 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
