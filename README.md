# Vera Compose Engine — magicpin AI Challenge

## Approach

A live FastAPI service (`bot/app.py`) exposing the 5 endpoints in
`challenge-testing-brief.md`, built as layered modules so decision quality
(which trigger to act on) is graded independently of message quality (what
it says):

- **`store.py`** — versioned in-memory context store, idempotent on
  `(scope, context_id, version)`, atomic replace under a lock.
- **`selection.py`** — pure Python, no LLM. Deterministic `urgency × recency
  × signal_match` scoring plus a hard eligibility gate (expiry, and consent
  gating for customer-scope triggers — a customer only gets messaged about
  things they actually opted into).
- **`voice.py`** — per-category tone/vocabulary/taboo checks compiled from
  `categories/*.json`, plus generic CTA-shape heuristics.
- **`grounding.py`** — the anti-hallucination gate. Every fact the composer
  states must resolve to a real path in the context bundle (a claims
  ledger), backed by a secondary stray-number scan across the whole bundle
  as a safety net for facts the LLM didn't self-report.
- **`composer.py`** — one LLM call (temperature=0) producing strict JSON,
  gated by grounding + voice + CTA validation before anything ships, with a
  deterministic, zero-LLM fallback template on any failure. `send_as` and
  `suppression_key` are never left to the LLM — both are mechanically
  derived from the inputs, since they're fully determined already.
- **`suppression.py`** — fired-key tracking scoped to `(recipient, key)`,
  not the bare key string (some suppression keys in the dataset don't embed
  a recipient id at all, e.g. a category-wide research digest).
- **`conversation.py`** — the `/v1/reply` state machine: auto-reply
  detection (content-pattern + 3x-repeat escalation), intent-transition
  routing, hostile/opt-out handling, anti-repetition, graceful exit after 3
  unresolved sends. Pure Python routing; content generation for an actual
  `send` delegates to `composer.compose_reply()`.

## Tradeoffs

- **Determinism is enforced by memoization, not by trusting the LLM.**
  Temperature=0 doesn't guarantee bit-identical output across separate API
  calls on most providers. `composer.py` caches by a hash of the full input
  bundle — identical inputs always return the literal same object for the
  life of the process, which is what "same inputs → same output" actually
  requires.
- **One shared prompt template**, parameterized by `CategoryContext.voice`,
  instead of a template per category or per trigger kind. Voice differences
  are already data; five near-duplicate templates would be designing for a
  need the dataset doesn't demonstrate.
- **Deterministic backstops over prompt compliance.** Rule-based instructions
  in the system prompt weren't reliable enough on their own (verified during
  self-critique against the real case studies — e.g. an explicit
  "never cite the internal id field" rule was still ignored on repeat
  calls). A regex catches any leaked internal id before it ships, same
  philosophy as the taboo-vocabulary and grounding checks.
- **Consent-scope mapping is conservative, not exhaustive.** Trigger kinds
  with clear evidence in the brief (`recall_due` → `recall_reminders`, etc.)
  are mapped explicitly; anything else just needs *some* non-empty consent
  rather than a guessed-at specific scope.
- **Restraint over reach.** `/v1/tick` caps to one action per
  `(merchant_id, customer_id)` pair per tick even when multiple triggers are
  eligible — testing-brief's own FAQ says restraint is rewarded.
- **No re-prompt/self-repair loop.** A failed LLM attempt goes straight to
  the fallback in the live service (a retry risks the judge's 30s budget for
  no guaranteed benefit); `scripts/generate_submission.py` *does* retry with
  real backoff, since that's an offline batch job with no such deadline.

## What additional context would have helped most

- **An exhaustive trigger-kind → consent-scope table.** The brief gives
  enough examples to infer the pattern but not a complete mapping — I had to
  choose a conservative default for kinds outside the evidenced set.
- **A resolution for two real contradictions between the two "source of
  truth" docs**, both logged in `problemFaced.txt`: whether URLs are ever
  allowed in a message body, and what the correct first-occurrence action is
  when a canned auto-reply is detected (testing-brief's own inline example
  disagrees with api-call-examples.md's explicitly-labeled "good" response
  for the identical input).
- **A production-tier LLM key during development**, not just a free-tier
  one. The free Groq key used here has a real ~8000-token/window rate limit
  that made calibrating live output quality slower than it needed to be —
  the fallback path handled it gracefully, but a paid key would have let
  more of the self-critique pass actually exercise the LLM path per hour of
  dev time.

Full log of every bug, ambiguity, and technology choice made along the way,
with reasoning, is in `problemFaced.txt`.
