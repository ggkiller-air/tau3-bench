"""Offline unit tests for the user-mode airline guards (SCHEMAFLEX_AIRLINE_GUARDS).

No vLLM / no network: `generate` is monkeypatched in llm_std_flex_agent, tool results
are fed as raw JSON strings. Covers:
  - AirlineGuardCache.ingest attribution (reservation / user / searches / flight status)
  - each of the 4 guards: >=1 pass + >=1 veto case (incl. the user-mode cancel_reason
    fail-closed simplification and the bags/book nonfree_baggages rewrite)
  - flag OFF => self._air is None, zero guard-path execution, write passes through
    untouched (retail non-regression: the only new branches are `self._air is not None`)
  - flag ON => veto->reprompt loop and the fail-closed exhaustion path

Run:  conda run -n tau3bench python tests/test_airline_flex_guards.py
(also pytest-compatible)
"""

import importlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tau2.agent.airline_guards import AirlineGuardCache  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage, UserMessage,
)

# --------------------------------------------------------------------------- #
# fixtures (mirrors tests/test_airline_armor.py)
# --------------------------------------------------------------------------- #

RES1 = {  # round_trip ATL->PHL->ATL, economy, no insurance, booked long ago
    "reservation_id": "RES1", "user_id": "mia_li_3668",
    "origin": "ATL", "destination": "PHL", "flight_type": "round_trip",
    "cabin": "economy", "insurance": "no", "created_at": "2024-05-01T10:00:00",
    "flights": [
        {"flight_number": "HAT001", "date": "2024-05-20", "origin": "ATL", "destination": "PHL", "price": 100},
        {"flight_number": "HAT002", "date": "2024-05-25", "origin": "PHL", "destination": "ATL", "price": 120},
    ],
    "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-01-01"}],
    "payment_history": [{"payment_id": "credit_card_1", "amount": 220}],
    "total_baggages": 1, "nonfree_baggages": 0, "status": None,
}

USER1 = {
    "user_id": "mia_li_3668", "membership": "silver",
    "payment_methods": {
        "credit_card_1": {"source": "credit_card", "id": "credit_card_1", "brand": "visa", "last_four": "1234"},
        "gift_card_1": {"source": "gift_card", "id": "gift_card_1", "amount": 100},
        "certificate_1": {"source": "certificate", "id": "certificate_1", "amount": 250},
        "certificate_2": {"source": "certificate", "id": "certificate_2", "amount": 100},
    },
}

SEARCH_ATL_PHL = [
    {"flight_number": "HAT010", "origin": "ATL", "destination": "PHL", "status": "available",
     "available_seats": {"economy": 9}, "prices": {"economy": 150}},
]
SEARCH_ONESTOP = [
    [
        {"flight_number": "HAT030", "origin": "ATL", "destination": "CLT", "date": "2024-05-21"},
        {"flight_number": "HAT031", "origin": "CLT", "destination": "PHL", "date": "2024-05-21"},
    ]
]


def _cache(**res_over):
    """Seeded cache: RES1 (+overrides) + USER1 + searches."""
    c = AirlineGuardCache()
    res = dict(RES1, **res_over)
    c.ingest("get_reservation_details", {"reservation_id": "RES1"}, json.dumps(res))
    c.ingest("get_user_details", {"user_id": "mia_li_3668"}, json.dumps(USER1))
    c.ingest("search_direct_flight", {}, json.dumps(SEARCH_ATL_PHL))
    c.ingest("search_onestop_flight", {}, json.dumps(SEARCH_ONESTOP))
    return c


def _tc(name, **arguments):
    return ToolCall(id="tc1", name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

def test_ingest():
    c = _cache()
    assert c.res_cache["RES1"]["cabin"] == "economy"
    assert c.user_cache["membership"] == "silver"
    assert c.flight_ep["HAT010"] == ("ATL", "PHL")
    assert c.flight_ep["HAT031"] == ("CLT", "PHL")     # onestop pair flattened
    assert c.flight_ep["HAT002"] == ("PHL", "ATL")     # endpoints from reservation JSON too
    c.ingest("get_flight_status", {"flight_number": "HAT001", "date": "2024-05-20"}, '"cancelled"')
    assert c.fstat[("HAT001", "2024-05-20")] == "cancelled"
    # a write result refreshes the reservation fact
    c.ingest("cancel_reservation", {"reservation_id": "RES1"},
             json.dumps(dict(RES1, status="cancelled")))
    assert c.res_cache["RES1"]["status"] == "cancelled"
    print("PASS test_ingest")


# --------------------------------------------------------------------------- #
# cancel guard (fail-closed; user-mode cancel_reason from user text)
# --------------------------------------------------------------------------- #

def test_cancel_guard():
    # veto: economy, old booking, uninsured, no reason -> not eligible
    assert _cache().guard(_tc("cancel_reservation", reservation_id="RES1")) is not None
    # veto: no cached reservation at all -> fail CLOSED
    assert AirlineGuardCache().guard(_tc("cancel_reservation", reservation_id="RES1")) is not None
    # veto: already cancelled
    assert _cache(status="cancelled").guard(_tc("cancel_reservation", reservation_id="RES1")) is not None
    # pass: business cabin
    assert _cache(cabin="business").guard(_tc("cancel_reservation", reservation_id="RES1")) is None
    # pass: booked within 24h of env clock
    assert _cache(created_at="2024-05-15T09:00:00").guard(
        _tc("cancel_reservation", reservation_id="RES1")) is None
    # insured + user clearly states a health reason -> pass
    c = _cache(insurance="yes")
    call = _tc("cancel_reservation", reservation_id="RES1")
    assert c.guard(call, "I got sick and my doctor says I cannot fly") is None
    # insured but NO clear health/weather mention -> treated as 'other' -> veto (fail-closed)
    assert c.guard(call, "I just changed my mind about the trip") is not None
    # uninsured + health mention -> still veto (insurance required)
    assert _cache().guard(call, "I am sick") is not None
    # verified airline-cancelled segment -> pass even when everything else fails
    c2 = _cache()
    c2.ingest("get_flight_status", {"flight_number": "HAT001", "date": "2024-05-20"}, '"cancelled"')
    assert c2.guard(call) is None
    # same-day segment already landed (fstat) -> not_flown False -> veto even for business
    c3 = _cache(cabin="business",
                flights=[{"flight_number": "HAT001", "date": "2024-05-15"}])
    c3.ingest("get_flight_status", {"flight_number": "HAT001", "date": "2024-05-15"}, '"landed"')
    assert c3.guard(call) is not None
    print("PASS test_cancel_guard")


# --------------------------------------------------------------------------- #
# update_reservation_flights guard (BE lock + segment chain)
# --------------------------------------------------------------------------- #

def test_update_flights_guard():
    same_flights = [{"flight_number": "HAT001", "date": "2024-05-20"},
                    {"flight_number": "HAT002", "date": "2024-05-25"}]
    # veto: basic economy with a changed flight set
    c = _cache(cabin="basic_economy")
    assert c.guard(_tc("update_reservation_flights", reservation_id="RES1",
                       cabin="basic_economy",
                       flights=[{"flight_number": "HAT010", "date": "2024-05-21"},
                                {"flight_number": "HAT002", "date": "2024-05-25"}],
                       payment_id="credit_card_1")) is not None
    # pass: basic economy, identical flight set (e.g. cabin/payment-only change)
    assert c.guard(_tc("update_reservation_flights", reservation_id="RES1",
                       cabin="economy", flights=same_flights,
                       payment_id="credit_card_1")) is None
    # veto: empty flights list (would wipe the reservation)
    assert _cache().guard(_tc("update_reservation_flights", reservation_id="RES1",
                              cabin="economy", flights=[],
                              payment_id="credit_card_1")) is not None
    # veto: broken chain — outbound replaced by CLT->PHL leg only (ATL origin violated)
    assert _cache().guard(_tc("update_reservation_flights", reservation_id="RES1",
                              cabin="economy",
                              flights=[{"flight_number": "HAT031", "date": "2024-05-21"},
                                       {"flight_number": "HAT002", "date": "2024-05-25"}],
                              payment_id="credit_card_1")) is not None
    # pass: valid replacement — onestop ATL->CLT->PHL + return PHL->ATL
    assert _cache().guard(_tc("update_reservation_flights", reservation_id="RES1",
                              cabin="economy",
                              flights=[{"flight_number": "HAT030", "date": "2024-05-21"},
                                       {"flight_number": "HAT031", "date": "2024-05-21"},
                                       {"flight_number": "HAT002", "date": "2024-05-25"}],
                              payment_id="credit_card_1")) is None
    # pass (fail-open): unknown flight number -> endpoints unprovable -> env decides
    assert _cache().guard(_tc("update_reservation_flights", reservation_id="RES1",
                              cabin="economy",
                              flights=[{"flight_number": "HAT999", "date": "2024-05-21"},
                                       {"flight_number": "HAT002", "date": "2024-05-25"}],
                              payment_id="credit_card_1")) is None
    print("PASS test_update_flights_guard")


# --------------------------------------------------------------------------- #
# book guard (pax cap + payment combo/balance + chain + baggage rewrite)
# --------------------------------------------------------------------------- #

def test_book_guard():
    pax1 = [{"first_name": "Mia", "last_name": "Li", "dob": "1990-01-01"}]

    def book(**over):
        kw = dict(user_id="mia_li_3668", origin="ATL", destination="PHL",
                  flight_type="one_way", cabin="economy",
                  flights=[{"flight_number": "HAT010", "date": "2024-05-21"}],
                  passengers=list(pax1),
                  payment_methods=[{"payment_id": "credit_card_1", "amount": 150}],
                  total_baggages=3, nonfree_baggages=3, insurance="no")
        kw.update(over)
        return _tc("book_reservation", **kw)

    # pass + rewrite: silver x economy => 2 free bags/pax; 3 total, 1 pax -> nonfree=1
    c = _cache()
    call = book()
    assert c.guard(call) is None
    assert call.arguments["nonfree_baggages"] == 1 and call.arguments["total_baggages"] == 3
    # veto: two certificates in one booking
    assert _cache().guard(book(payment_methods=[
        {"payment_id": "certificate_1", "amount": 100},
        {"payment_id": "certificate_2", "amount": 50}])) is not None
    # veto: gift card overdraw (balance 100)
    assert _cache().guard(book(payment_methods=[
        {"payment_id": "gift_card_1", "amount": 150}])) is not None
    # veto: payment id not in the cached user profile
    assert _cache().guard(book(payment_methods=[
        {"payment_id": "credit_card_999", "amount": 150}])) is not None
    # veto: more than five passengers
    assert _cache().guard(book(passengers=pax1 * 6)) is not None
    # veto: chain broken vs args endpoints (LAX origin but ATL->PHL flight)
    assert _cache().guard(book(origin="LAX", destination="PHL")) is not None
    # pass (fail-open): user profile never observed -> payment checks skipped
    empty = AirlineGuardCache()
    assert empty.guard(book(payment_methods=[
        {"payment_id": "certificate_1", "amount": 100},
        {"payment_id": "certificate_2", "amount": 50}])) is None
    print("PASS test_book_guard")


# --------------------------------------------------------------------------- #
# bags guard (only-increase + nonfree rewrite)
# --------------------------------------------------------------------------- #

def test_bags_guard():
    def bags(tb, nb):
        return _tc("update_reservation_baggages", reservation_id="RES1",
                   total_baggages=tb, nonfree_baggages=nb, payment_id="credit_card_1")
    # pass + rewrite: RES1 silver x economy x 1 pax => 2 free; 3 total -> nonfree=1
    c = _cache()
    call = bags(3, 0)  # 8B under-declared nonfree; guard rewrites, not vetoes
    assert c.guard(call) is None
    assert call.arguments["nonfree_baggages"] == 1
    # veto: removing bags (1 -> 0)
    assert _cache().guard(bags(0, 0)) is not None
    # pass (fail-open): reservation never observed
    assert AirlineGuardCache().guard(bags(0, 0)) is None
    print("PASS test_bags_guard")


# --------------------------------------------------------------------------- #
# agent integration: flag gating, veto loop, exhaustion, pass-through
# --------------------------------------------------------------------------- #

def _load(on: bool):
    os.environ["SCHEMAFLEX_AIRLINE_GUARDS"] = "1" if on else ""
    import tau2.agent.llm_std_flex_agent as mod
    mod = importlib.reload(mod)
    assert mod._AIRGUARD_ON is on
    return mod


def _agent(mod):
    return mod.LLMStdFlexAgent(tools=[], domain_policy="policy",
                               llm="openai/stub", llm_args={})


def _state(msgs):
    return SimpleNamespace(system_messages=[SystemMessage(role="system", content="sys")],
                           messages=list(msgs))


def _patch_gen(mod, outputs):
    """mod.generate stub returning canned messages in order; records call_names."""
    calls = []
    def fake(model=None, tools=None, messages=None, call_name=None, **kw):
        calls.append(call_name)
        return outputs[min(len(calls) - 1, len(outputs) - 1)]
    mod.generate = fake
    return calls


def test_flag_off_zero_execution():
    mod = _load(False)
    ag = _agent(mod)
    assert ag._air is None  # no cache object exists at all when the flag is off

    # a poisoned cache class proves no guard path could run even if reached
    class Boom:
        def __getattr__(self, _):
            raise AssertionError("airline guard path executed with flag OFF")
    mod.AirlineGuardCache = Boom

    doomed = AssistantMessage(role="assistant", content=None, tool_calls=[
        _tc("cancel_reservation", reservation_id="RES1")])
    calls = _patch_gen(mod, [doomed])
    st = _state([UserMessage(role="user", content="cancel my reservation RES1")])
    out = ag._generate_next_message(None, st)
    # OFF: the (vetoable!) airline write passes through untouched, single LLM call,
    # arguments unmodified => behavior identical to the pre-change agent (and retail
    # runs never enter any new branch: all are behind `self._air is not None`).
    assert out is doomed and calls == ["agent_response"]
    assert out.tool_calls[0].arguments == {"reservation_id": "RES1"}
    print("PASS test_flag_off_zero_execution")


def test_flag_on_veto_then_text():
    mod = _load(True)
    ag = _agent(mod)
    assert isinstance(ag._air, mod.AirlineGuardCache)
    doomed = AssistantMessage(role="assistant", content=None, tool_calls=[
        _tc("cancel_reservation", reservation_id="RES1")])  # nothing cached -> fail closed
    sorry = AssistantMessage(role="assistant", content="I cannot cancel that reservation.")
    calls = _patch_gen(mod, [doomed, sorry])
    st = _state([UserMessage(role="user", content="cancel RES1 now")])
    out = ag._generate_next_message(None, st)
    # veto -> ONE reprompt (agent_guard_prompt) -> its text answer is returned;
    # the doomed write never surfaces.
    assert calls == ["agent_response", "agent_guard_prompt"]
    assert out is sorry and not out.is_tool_call()
    print("PASS test_flag_on_veto_then_text")


def test_flag_on_exhaustion_fail_closed():
    mod = _load(True)
    ag = _agent(mod)
    doomed = AssistantMessage(role="assistant", content=None, tool_calls=[
        _tc("cancel_reservation", reservation_id="RES1")])
    calls = _patch_gen(mod, [doomed])  # the 8B insists on the doomed write forever
    st = _state([UserMessage(role="user", content="cancel RES1")])
    out = ag._generate_next_message(None, st)
    assert calls == ["agent_response"] + ["agent_guard_prompt"] * mod._AIRGUARD_MAX_VETO
    assert not out.is_tool_call() and "cannot perform that action" in (out.content or "")
    print("PASS test_flag_on_exhaustion_fail_closed")


def test_flag_on_ingest_and_pass_with_rewrite():
    mod = _load(True)
    ag = _agent(mod)
    # feed raw tool results through the real message path (attribution via tool_call ids)
    prior = AssistantMessage(role="assistant", content=None, tool_calls=[
        ToolCall(id="c_res", name="get_reservation_details", arguments={"reservation_id": "RES1"}),
        ToolCall(id="c_usr", name="get_user_details", arguments={"user_id": "mia_li_3668"}),
    ])
    tm1 = ToolMessage(id="c_res", role="tool", content=json.dumps(RES1), requestor="assistant")
    tm2 = ToolMessage(id="c_usr", role="tool", content=json.dumps(USER1), requestor="assistant")
    proposal = AssistantMessage(role="assistant", content=None, tool_calls=[
        _tc("update_reservation_baggages", reservation_id="RES1",
            total_baggages=3, nonfree_baggages=0, payment_id="credit_card_1")])
    _patch_gen(mod, [proposal])
    st = _state([prior])
    out = ag._generate_next_message(
        MultiToolMessage(role="tool", tool_messages=[tm1, tm2]), st)
    assert ag._air.res_cache["RES1"]["reservation_id"] == "RES1"
    assert ag._air.user_cache["membership"] == "silver"
    # guard passed and deterministically rewrote nonfree_baggages (2 free/pax, silver)
    assert out is proposal
    assert out.tool_calls[0].arguments["nonfree_baggages"] == 1
    print("PASS test_flag_on_ingest_and_pass_with_rewrite")


if __name__ == "__main__":
    test_ingest()
    test_cancel_guard()
    test_update_flights_guard()
    test_book_guard()
    test_bags_guard()
    test_flag_off_zero_execution()
    test_flag_on_veto_then_text()
    test_flag_on_exhaustion_fail_closed()
    test_flag_on_ingest_and_pass_with_rewrite()
    print("ALL PASS")
