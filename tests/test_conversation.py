"""Tests conversation.py's deterministic routing. bot.conversation.compose_reply
is monkeypatched throughout — these must run without an API key or network
access, same as test_composer.py.
"""

import pytest

from bot import conversation
from bot.conversation import ConversationStore, handle_reply
from bot.models import ComposedMessage

AUTO_REPLY_TEXT = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."


def _new_state(store, conv_id="conv_1"):
    return store.start(conv_id, "m_001", None, "dentists", "trg_1", "vera", "Initial nudge")


def test_hostile_message_ends_immediately():
    store = ConversationStore()
    state = _new_state(store)
    result = handle_reply(state, "Stop messaging me. This is useless spam.", {}, {"identity": {}}, {"payload": {}}, None)
    assert result["action"] == "end"
    assert state.ended


def test_ended_conversation_stays_ended():
    store = ConversationStore()
    state = _new_state(store)
    state.ended = True
    result = handle_reply(state, "actually wait, tell me more", {}, {"identity": {}}, {"payload": {}}, None)
    assert result["action"] == "end"


def test_auto_reply_escalation_matches_send_wait_end():
    store = ConversationStore()
    state = _new_state(store)

    r1 = handle_reply(state, AUTO_REPLY_TEXT, {}, {"identity": {}}, {"payload": {}}, None)
    assert r1["action"] == "send"

    r2 = handle_reply(state, AUTO_REPLY_TEXT, {}, {"identity": {}}, {"payload": {}}, None)
    assert r2["action"] == "wait"
    assert r2["wait_seconds"] == 86400

    r3 = handle_reply(state, AUTO_REPLY_TEXT, {}, {"identity": {}}, {"payload": {}}, None)
    assert r3["action"] == "end"
    assert state.ended


def test_intent_transition_skips_qualification(monkeypatch):
    fake_reply = ComposedMessage(
        body="Drafting your patient WhatsApp now.", cta="binary_confirm_cancel",
        send_as="vera", suppression_key="sk", rationale="Explicit commitment detected.",
    )
    monkeypatch.setattr(conversation, "compose_reply", lambda **kwargs: fake_reply)

    store = ConversationStore()
    state = _new_state(store)
    state.send_count = 2  # simulate 2 prior qualifying turns

    result = handle_reply(state, "Ok, let's do it. What's next?", {}, {"identity": {}}, {"payload": {}}, None)
    assert result["action"] == "send"
    assert result["body"] == "Drafting your patient WhatsApp now."


def test_max_sends_without_resolution_ends_conversation(monkeypatch):
    fake_reply = ComposedMessage(
        body="Following up again.", cta="open_ended", send_as="vera", suppression_key="sk", rationale="x",
    )
    monkeypatch.setattr(conversation, "compose_reply", lambda **kwargs: fake_reply)

    store = ConversationStore()
    state = _new_state(store)
    state.send_count = conversation.MAX_SENDS_WITHOUT_RESOLUTION

    result = handle_reply(state, "hmm, not sure yet", {}, {"identity": {}}, {"payload": {}}, None)
    assert result["action"] == "end"
    assert state.ended


def test_anti_repetition_ends_instead_of_resending_verbatim(monkeypatch):
    fake_reply = ComposedMessage(
        body="Initial nudge", cta="open_ended", send_as="vera", suppression_key="sk", rationale="x",
    )
    monkeypatch.setattr(conversation, "compose_reply", lambda **kwargs: fake_reply)

    store = ConversationStore()
    state = _new_state(store)  # sent_bodies already contains "Initial nudge"

    result = handle_reply(state, "some ordinary reply", {}, {"identity": {}}, {"payload": {}}, None)
    assert result["action"] == "end"
    assert "repeated" in result["rationale"].lower()
