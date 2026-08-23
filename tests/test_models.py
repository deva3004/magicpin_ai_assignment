"""Parses the real dataset through every context model — the concrete proof
that extra="allow" handles the dataset's undocumented fields, and that the
two keyword-collision aliases (ConversationTurn.from_, VoiceProfile.register_)
actually round-trip correctly.
"""

import json
from pathlib import Path

from bot.models import CategoryContext, CustomerContext, MerchantContext, TriggerContext

DATASET = Path(__file__).parent.parent / "dataset"


def test_all_category_seeds_parse():
    for f in (DATASET / "categories").glob("*.json"):
        CategoryContext(**json.load(open(f)))


def test_all_merchant_seeds_parse():
    for m in json.load(open(DATASET / "merchants_seed.json"))["merchants"]:
        MerchantContext(**m)


def test_all_trigger_seeds_parse():
    for t in json.load(open(DATASET / "triggers_seed.json"))["triggers"]:
        TriggerContext(**t)


def test_all_customer_seeds_parse():
    for c in json.load(open(DATASET / "customers_seed.json"))["customers"]:
        CustomerContext(**c)


def test_merchant_conversation_history_from_alias():
    payload = json.load(open(DATASET / "merchants_seed.json"))["merchants"][0]
    merchant = MerchantContext(**payload)
    assert merchant.conversation_history[0].from_ == "vera"


def test_voice_register_alias():
    category = CategoryContext(**json.load(open(DATASET / "categories" / "dentists.json")))
    assert category.voice.register_ == "respectful_collegial"


def test_extra_fields_preserved_not_rejected():
    # review_themes isn't a documented field on MerchantContext but real data has it
    payload = json.load(open(DATASET / "merchants_seed.json"))["merchants"][0]
    merchant = MerchantContext(**payload)
    dumped = merchant.model_dump()
    assert "review_themes" in dumped
