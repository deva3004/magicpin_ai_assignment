"""Per-category voice constraints, compiled from categories/*.json, plus
generic (non-category-specific) CTA-shape checks.

Two things live here:

  1. load_voice() — pull the `voice` block out of a CategoryContext payload
     into a typed VoiceProfile (models.py already defines the shape; this
     module is the "compile it and give me a checker" layer CLAUDE.md's
     architecture calls for).

  2. find_taboo_violations() — a deterministic, case-insensitive scan of a
     composed body against that category's vocab_taboo list. This is a hard,
     well-defined check (literal match against real per-category data) — the
     tone counterpart to grounding.py's fact checks.

     Real data quirk: dentists.json's taboo list includes "FDA-approved (use
     only when actually applicable)" — a usage caveat baked into the taboo
     string itself. Matched literally that phrase would never appear in any
     real message, silently disabling the one legally-sensitive entry. So
     trailing "(...)" qualifiers are stripped before matching.

Note on scope: CLAUDE.md's original Phase 1 checklist for this module says
"load per-category voice/vocab/taboo/cta-defaults from categories/*.json."
There is no "cta_defaults" field anywhere in the actual category JSON
(checked all 5) — CTA rules (single primary CTA, CTA lands in the last
sentence) are cross-category policy from challenge-brief.md §5 and §11, not
per-category data. So count_ctas()/has_buried_cta() below are generic text
heuristics, not compiled from any category-specific field. Logged in
problemFaced.txt.

Explicitly out of scope: verifying a message actually honors a language
preference (e.g. hi-en code-mix). Reliable Hindi/Hinglish detection from
Latin-script text isn't a cheap deterministic check the way taboo-vocab
matching is — that's steered via the composer's prompt instructions, not
validated post-hoc here.
"""

from __future__ import annotations

import re
from typing import Any

from bot.models import VoiceProfile

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PAREN_NOTE_RE = re.compile(r"\s*\([^)]*\)\s*$")


def load_voice(category_payload: dict[str, Any]) -> VoiceProfile:
    return VoiceProfile(**category_payload.get("voice", {}))


def find_taboo_violations(body: str, voice: VoiceProfile) -> list[str]:
    lowered = body.lower()
    hits = []
    for phrase in voice.vocab_taboo:
        clean = _PAREN_NOTE_RE.sub("", phrase).strip()
        if clean and clean.lower() in lowered:
            hits.append(phrase)
    return hits


def count_ctas(body: str) -> int:
    """Counts '?' characters as a rough proxy for distinct asks.

    NOT a reliable "exactly one CTA = exactly one '?'" rule — case-studies.md's
    own gold examples disprove that (Case Study 2's CTA, "Reply 1 for Wed, 2
    for Thu, or tell us a time that works.", has zero question marks; it's an
    imperative-style CTA, not a question). composer.py only uses this as an
    upper-bound check (2+ questions is still a real multi-CTA red flag that
    doesn't appear in any gold example), never as an exact-match requirement.
    """
    return body.count("?")


def has_buried_cta(body: str) -> bool:
    """True if a '?' exists but substantial content follows it (not just a short
    trailing source citation like '— JIDA Oct 2026 p.14', which several gold
    examples legitimately have after the question mark).

    NOT used as a hard validation gate in composer.py — Case Study 4's gold
    CTA is a question in the MIDDLE of the message followed by two more
    sentences of legitimate elaboration, which this function would (correctly,
    by its own definition) flag as "buried." Kept available for advisory/
    manual-review use, not enforcement.
    """
    if "?" not in body:
        return False
    tail = body[body.rindex("?") + 1 :].strip()
    if not tail or tail.startswith(("—", "-", "(")):
        return False
    return len(tail) > 50
