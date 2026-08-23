"""Fact-ledger validator — the hard gate between an LLM composition and the wire.

Two independent checks, both pure functions of (body, claims, context_bundle):

  1. Claim resolution — composer.py's LLM call is expected to emit a small
     "claims" ledger alongside the message body: each verifiable fact paired
     with the dotted path in the context bundle it came from (e.g.
     "category.digest[0].trial_n"). Every declared claim must resolve to a
     real value at that path, or it's fabrication by definition — this is
     the ONLY fully reliable signal, since it's checking the LLM's own
     stated provenance rather than guessing.

  2. Stray-number scan — a secondary, best-effort net for numbers the LLM
     used but never declared as a claim. It flattens the entire context
     bundle (walking into string fields too, since most facts live inside
     digest/summary/title prose, not as bare numeric leaves) into a set of
     every number that legitimately appears anywhere, then flags any
     "significant" number in the body that isn't in that set. "Significant"
     deliberately excludes bare integers <= 20 with no currency/percent/comma
     formatting — those are almost always structural (CTA option indices like
     "Reply 1 for Wed, 2 for Thu", generic counts like "2 slots" or "5 min")
     rather than a claimed fact, and case-studies.md's own good examples are
     full of them. Anything with a % sign, a ₹ sign, comma grouping, or a
     decimal point is treated as significant regardless of size, because
     that's the shape every genuine fabricated stat in the rubric's failure
     examples takes ("22%", "₹299", "2,100-patient", "62.5%"). This threshold
     is a judgment call, logged in problemFaced.txt.

Callers should include `now` (the current simulated time from /v1/tick) in
the bundle they pass in, so date-derived phrasing doesn't get falsely
flagged as a stray number.

This module only *checks* — it doesn't decide what happens on failure. That
policy (retry the LLM, fall back to the deterministic template, etc.)
belongs to composer.py, per CLAUDE.md's module split: grounding.py stays
pure and independently unit-testable.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

NUMBER_RE = re.compile(r"₹?\d[\d,]*(?:\.\d+)?%?")
IGNORABLE_INT_CEILING = 20


class Claim(BaseModel):
    text: str
    source_path: str


class GroundingResult(BaseModel):
    claims_ok: bool
    unresolved_claims: list[Claim]
    stray_numbers: list[str]

    @property
    def ok(self) -> bool:
        return self.claims_ok and not self.stray_numbers


def resolve_path(bundle: dict[str, Any], path: str) -> tuple[bool, Any]:
    """Walk a dotted path like 'category.digest[0].trial_n' through the bundle."""
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    node: Any = bundle
    for tok in tokens:
        if tok.startswith("["):
            idx = int(tok[1:-1])
            if not isinstance(node, list) or idx >= len(node):
                return False, None
            node = node[idx]
        else:
            if not isinstance(node, dict) or tok not in node:
                return False, None
            node = node[tok]
    return True, node


def _normalize_number(token: str) -> tuple[str, bool]:
    """Returns (normalized_digits, is_significant)."""
    has_formatting = "%" in token or "₹" in token or "," in token or "." in token
    digits = token.strip().lstrip("₹").rstrip("%").replace(",", "")
    if has_formatting:
        significant = True
    else:
        try:
            significant = abs(int(digits)) > IGNORABLE_INT_CEILING
        except ValueError:
            significant = True  # unexpected shape — err toward flagging, not hiding
    return digits, significant


def _collect_known_numbers(node: Any, out: set[str]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect_known_numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_known_numbers(v, out)
    elif isinstance(node, str):
        for match in NUMBER_RE.findall(node):
            out.add(_normalize_number(match)[0])
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(_normalize_number(str(node))[0])


def check_grounding(body: str, claims: list[Claim], bundle: dict[str, Any]) -> GroundingResult:
    unresolved: list[Claim] = []
    known: set[str] = set()
    _collect_known_numbers(bundle, known)

    for claim in claims:
        found, value = resolve_path(bundle, claim.source_path)
        if not found:
            unresolved.append(claim)
        else:
            known.add(_normalize_number(str(value))[0])

    stray: list[str] = []
    for match in NUMBER_RE.findall(body):
        digits, significant = _normalize_number(match)
        if significant and digits not in known:
            stray.append(match)

    return GroundingResult(claims_ok=not unresolved, unresolved_claims=unresolved, stray_numbers=stray)
