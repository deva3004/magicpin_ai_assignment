import json
from pathlib import Path

from bot.grounding import Claim, check_grounding, resolve_path

CATEGORY = json.loads(Path("dataset/categories/dentists.json").read_text())


def test_resolve_path_dict_and_list_traversal():
    bundle = {"category": CATEGORY}
    found, value = resolve_path(bundle, "category.digest[0].trial_n")
    assert found
    assert value == 2100


def test_resolve_path_missing_returns_false():
    bundle = {"category": CATEGORY}
    found, value = resolve_path(bundle, "category.digest[99].trial_n")
    assert not found
    found2, _ = resolve_path(bundle, "category.nonexistent_field")
    assert not found2


def test_gold_case_study_1_message_passes_clean():
    bundle = {"category": CATEGORY}
    body = (
        "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult "
        "patients — 2,100-patient trial showed 3-month fluoride recall cuts caries "
        "recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me "
        "to pull it + draft a patient-ed WhatsApp you can share?  — JIDA Oct 2026 p.14"
    )
    claims = [Claim(text="2100-patient trial, 38% figure", source_path="category.digest[0]")]
    result = check_grounding(body, claims, bundle)
    assert result.ok
    assert result.claims_ok
    assert result.stray_numbers == []


def test_fabricated_stats_are_flagged_as_stray():
    bundle = {"category": CATEGORY}
    body = "Fake stat: 71 percent of patients love us, per a JIDA March 2019 report."
    result = check_grounding(body, [], bundle)
    assert not result.ok
    assert "71" in result.stray_numbers
    assert "2019" in result.stray_numbers


def test_unresolved_claim_fails_claims_ok():
    bundle = {"category": CATEGORY}
    claims = [Claim(text="made up", source_path="category.digest[0].nonexistent_field")]
    result = check_grounding("some text with no numbers", claims, bundle)
    assert not result.claims_ok
    assert not result.ok


def test_small_bare_integers_are_not_flagged():
    bundle = {"category": CATEGORY}
    # "2 slots", "5 min", "Reply 1 for Wed, 2 for Thu" style numbers — none of
    # these are in the context, but they're all <= 20 with no currency/percent/
    # comma/decimal formatting, so they shouldn't be treated as claims.
    body = "Reply 1 for Wed, 2 for Thu. Takes about 5 min, 2 slots available."
    result = check_grounding(body, [], bundle)
    assert result.stray_numbers == []


def test_formatted_numbers_are_flagged_even_when_small():
    # Isolated fixture, not the real CATEGORY — dentists.json coincidentally
    # contains a standalone "5" (in the product name "Trios 5"), which would
    # make this assertion depend on incidental real-data noise rather than
    # the actual behavior under test.
    bundle = {"category": {"slug": "dentists"}}
    body = "Only ₹5 today!"
    result = check_grounding(body, [], bundle)
    assert "₹5" in result.stray_numbers
