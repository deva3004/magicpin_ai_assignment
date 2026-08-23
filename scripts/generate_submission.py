#!/usr/bin/env python3
"""Runs compose() over all 30 canonical test pairs (dataset_expanded/test_pairs.json)
-> submission.jsonl, per challenge-brief.md §7.2.

Retries with real backoff on a failed/rejected LLM attempt, calling
composer.py's internal building blocks directly (_build_bundle,
_try_llm_compose, _compose_fallback) rather than the public compose(). Two
reasons this lives here and not in composer.py itself:

  1. compose() is memoized for the live service's determinism guarantee —
     calling it again with the same inputs after a failure would just replay
     the cached fallback, never actually retrying the LLM.
  2. composer.py deliberately has no retry loop, because the live service is
     bound by the judge's 30s-per-call budget (testing-brief §5) and a retry
     isn't guaranteed to fit in what's left of it. This script has no such
     deadline — it's an offline batch job, so spending real time on a real
     retry here is a completely different, reasonable tradeoff. (Confirmed
     during composer.py's self-critique that this Groq key's rate-limit
     window resets in ~10-25s, hence the retry delay below.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `bot` is importable when run directly

from bot.composer import _build_bundle, _compose_fallback, _try_llm_compose
from bot.models import ComposedMessage
from bot.voice import load_voice

DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_RETRY_DELAY_SECONDS = 25.0
DEFAULT_MAX_ATTEMPTS = 3


def compose_with_retries(category, merchant, trigger, customer, retries: int, retry_delay: float) -> ComposedMessage:
    send_as = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get("suppression_key", "")
    voice = load_voice(category)
    bundle = _build_bundle(category, merchant, trigger, customer)

    for attempt in range(retries):
        result = _try_llm_compose(voice, bundle, send_as, suppression_key)
        if result is not None:
            return result
        if attempt < retries - 1:
            print(f"    attempt {attempt + 1}/{retries} failed/rejected — retrying in {retry_delay:.0f}s...")
            time.sleep(retry_delay)

    body, cta = _compose_fallback(merchant, trigger, customer)
    return ComposedMessage(
        body=body, cta=cta, send_as=send_as, suppression_key=suppression_key,
        rationale=f"Deterministic fallback template — LLM failed validation or was unavailable after {retries} attempts.",
    )


def load_dataset(data_dir: Path):
    categories = {json.load(open(f))["slug"]: json.load(open(f)) for f in (data_dir / "categories").glob("*.json")}
    merchants = {json.load(open(f))["merchant_id"]: json.load(open(f)) for f in (data_dir / "merchants").glob("*.json")}
    customers = {json.load(open(f))["customer_id"]: json.load(open(f)) for f in (data_dir / "customers").glob("*.json")}
    triggers = {json.load(open(f))["id"]: json.load(open(f)) for f in (data_dir / "triggers").glob("*.json")}
    return categories, merchants, customers, triggers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset_expanded")
    parser.add_argument("--out", default="submission.jsonl")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="delay between pairs")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    categories, merchants, customers, triggers = load_dataset(data_dir)
    test_pairs = json.load(open(data_dir / "test_pairs.json"))["pairs"]

    lines: list[dict] = []
    for i, pair in enumerate(test_pairs):
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None

        if trigger is None or merchant is None:
            print(f"[{pair['test_id']}] SKIPPED — missing trigger or merchant in dataset")
            continue

        category = categories.get(merchant.get("category_slug", ""))
        if category is None:
            print(f"[{pair['test_id']}] SKIPPED — missing category {merchant.get('category_slug')!r}")
            continue

        composed = compose_with_retries(category, merchant, trigger, customer, args.max_attempts, args.retry_delay)
        used_fallback = "fallback" in composed.rationale.lower()
        print(f"[{pair['test_id']}] {'FALLBACK' if used_fallback else 'LLM   '} -> {composed.body[:70]}")

        lines.append({
            "test_id": pair["test_id"],
            "body": composed.body,
            "cta": composed.cta,
            "send_as": composed.send_as,
            "suppression_key": composed.suppression_key,
            "rationale": composed.rationale,
        })

        if i < len(test_pairs) - 1:
            time.sleep(args.delay)

    with open(args.out, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    n_fallback = sum(1 for l in lines if "fallback" in l["rationale"].lower())
    print(f"\nWrote {len(lines)} lines to {args.out} ({len(lines) - n_fallback} via LLM, {n_fallback} via fallback)")


if __name__ == "__main__":
    main()
