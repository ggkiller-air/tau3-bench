"""Offline unit tests for the airline ARMOR hooks in schema_solo_std_agent.py.

No vLLM / no network: agents are built via object.__new__ (skipping __init__'s LLM
machinery), task_state and the raw-JSON caches are mocked, and the base
SchemaSoloAgent.generate_next_message is monkeypatched for the veto-loop tests.

Run:  conda run -n tau3bench python tests/test_airline_armor.py
(also pytest-compatible; each test_* function is standalone)

Covers, per hook: >=1 pass case + >=1 veto case, plus:
  - flag OFF => zero armor-path execution (proven by an instance that HAS NO armor
    attributes at all: any armor code touching self._res_cache etc. would raise)
  - cancel guard's insurance + STALE cancel_reason fail-closed case (amendment 4)
  - the ADVANCE-sentinel epoch tracker
  - the veto->error-feedback->retry loop and its fail-closed exhaustion path
"""

import importlib
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --------------------------------------------------------------------------- #
# fixtures
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

SEARCH_ATL_PHL = [  # search_direct_flight ATL->PHL 2024-05-21
    {"flight_number": "HAT010", "origin": "ATL", "destination": "PHL", "status": "available",
     "scheduled_departure_time_est": "08:00:00", "scheduled_arrival_time_est": "10:00:00",
     "available_seats": {"economy": 9}, "prices": {"economy": 150}},
]
SEARCH_LAX_JFK = [
    {"flight_number": "HAT020", "origin": "LAX", "destination": "JFK", "status": "available",
     "scheduled_departure_time_est": "08:00:00", "scheduled_arrival_time_est": "16:00:00",
     "available_seats": {"economy": 9}, "prices": {"economy": 300}},
]
SEARCH_ONESTOP = [  # search_onestop_flight ATL->PHL via CLT
    [
        {"flight_number": "HAT030", "origin": "ATL", "destination": "CLT", "date": "2024-05-21"},
        {"flight_number": "HAT031", "origin": "CLT", "destination": "PHL", "date": "2024-05-21"},
    ]
]


def _load(armor_on: bool):
    """(Re)import the module under the requested flag value."""
    os.environ["SCHEMAFLEX_AIRLINE_ARMOR"] = "1" if armor_on else ""
    import tau2.agent.schema_solo_std_agent as mod
    mod = importlib.reload(mod)
    assert mod._ARMOR_ON is armor_on
    return mod


def _agent(mod):
    """Armed agent skeleton: no LLM, no schema — only the armor state."""
    ag = object.__new__(mod.SchemaSoloStdAgent)
    ag._armor_reset()
    ag.task = SimpleNamespace(id="test-task")
    return ag


def _tm(mod, call_id, content):
    return mod.ToolMessage(id=call_id, role="tool", content=content, requestor="assistant")


def _seed_caches(mod, ag):
    """Ingest RES1 / USER1 / two direct searches / one onestop search as raw ToolMessages."""
    ag._call_args["c_res"] = ("get_reservation_details", {"reservation_id": "RES1"})
    ag._call_args["c_usr"] = ("get_user_details", {"user_id": "mia_li_3668"})
    ag._call_args["c_s1"] = ("search_direct_flight", {"origin": "ATL", "destination": "PHL", "date": "2024-05-21"})
    ag._call_args["c_s2"] = ("search_direct_flight", {"origin": "LAX", "destination": "JFK", "date": "2024-05-21"})
    ag._call_args["c_s3"] = ("search_onestop_flight", {"origin": "ATL", "destination": "PHL", "date": "2024-05-21"})
    ag._armor_ingest(_tm(mod, "c_res", json.dumps(RES1)))
    ag._armor_ingest(_tm(mod, "c_usr", json.dumps(USER1)))
    ag._armor_ingest(_tm(mod, "c_s1", json.dumps(SEARCH_ATL_PHL)))
    ag._armor_ingest(_tm(mod, "c_s2", json.dumps(SEARCH_LAX_JFK)))
    ag._armor_ingest(_tm(mod, "c_s3", json.dumps(SEARCH_ONESTOP)))
    assert ag._res_cache["RES1"]["cabin"] == "economy"
    assert ag._user_cache["membership"] == "silver"
    assert ag._flight_ep["HAT010"] == ("ATL", "PHL")
    assert ag._flight_ep["HAT031"] == ("CLT", "PHL")  # onestop pair got flattened
    assert ag._flight_ep["HAT002"] == ("PHL", "ATL")  # endpoints also from the reservation JSON


def _call(mod, name, **arguments):
    from tau2.data_model.message import ToolCall
    return ToolCall(id="tc1", name=name, arguments=arguments)


# --------------------------------------------------------------------------- #
# 0) flag OFF => zero armor-path execution
# --------------------------------------------------------------------------- #

def test_off_is_passthrough():
    mod = _load(False)
    ag = object.__new__(mod.SchemaSoloStdAgent)  # NO armor attributes exist at all
    ag.task = SimpleNamespace(id="t")
    marker_calls = []
    vetoable = mod.AssistantMessage(
        role="assistant", content=None,
        tool_calls=[_call(mod, "cancel_reservation", reservation_id="NOPE")])

    orig = mod.SchemaSoloAgent.generate_next_message
    def stub(self, message, state):
        marker_calls.append(message)
        return vetoable, state
    mod.SchemaSoloAgent.generate_next_message = stub
    try:
        state = SimpleNamespace(task_state={}, messages=[], finished=False)
        out, _ = mod.SchemaSoloStdAgent.generate_next_message(ag, None, state)
    finally:
        mod.SchemaSoloAgent.generate_next_message = orig
    # OFF: the (vetoable!) call passes through untouched, exactly one base call, and no
    # armor attribute was ever touched (the instance has none — any touch would raise).
    assert out is vetoable and len(marker_calls) == 1
    # _emit_done OFF path: plain stop, no comm diversion even with a pending note
    state2 = SimpleNamespace(task_state={"resolved": True, "comm_note": "4"}, messages=[], finished=False)
    msg = mod.SchemaSoloStdAgent._emit_done(ag, state2)
    assert msg.content == mod.SchemaSoloStdAgent.STOP_TOKEN
    print("PASS test_off_is_passthrough")


# --------------------------------------------------------------------------- #
# A) comm_note emission
# --------------------------------------------------------------------------- #

def test_comm_note():
    mod = _load(True)
    ag = _agent(mod)
    # pass case: resolved + note -> ONE final message carrying note AND stop token
    st = SimpleNamespace(task_state={"resolved": True, "comm_note": "your gift cards total 327"},
                         messages=[], finished=False)
    msg = ag._emit_done(st)
    assert "327" in msg.content and mod.SchemaSoloStdAgent.STOP_TOKEN in msg.content
    assert mod.SchemaSoloStdAgent.is_stop(msg)  # substring stop detection -> AGENT_STOP
    assert ag._comm_emitted is True
    # emitted once: a second stop is bare
    msg2 = ag._emit_done(st)
    assert msg2.content == mod.SchemaSoloStdAgent.STOP_TOKEN
    # veto-side case: NOT resolved -> note suppressed (only the resolved branch reports)
    ag2 = _agent(mod)
    st2 = SimpleNamespace(task_state={"resolved": False, "comm_note": "4"}, messages=[], finished=False)
    assert ag2._emit_done(st2).content == mod.SchemaSoloStdAgent.STOP_TOKEN
    # resolved but empty/null note -> bare stop
    ag3 = _agent(mod)
    st3 = SimpleNamespace(task_state={"resolved": True, "comm_note": None}, messages=[], finished=False)
    assert ag3._emit_done(st3).content == mod.SchemaSoloStdAgent.STOP_TOKEN
    print("PASS test_comm_note")


# --------------------------------------------------------------------------- #
# B) update_reservation_flights pre-guard
# --------------------------------------------------------------------------- #

def test_update_flights_guard():
    mod = _load(True)
    ag = _agent(mod)
    _seed_caches(mod, ag)
    new_flights = [
        {"flight_number": "HAT010", "date": "2024-05-21", "origin": "ATL", "destination": "PHL"},
        {"flight_number": "HAT002", "date": "2024-05-25", "origin": "PHL", "destination": "ATL"},
    ]
    ts = {"new_flights": new_flights, "res_origin": "ATL", "res_destination": "PHL",
          "res_trip_type": "round_trip"}
    ok_args = {"reservation_id": "RES1", "cabin": "economy",
               "flights": [dict(f) for f in new_flights], "payment_id": "credit_card_1"}
    # pass: verbatim match, endpoints from search+reservation caches, chain closes ATL->PHL->ATL
    assert ag._armor_guard(_call(mod, "update_reservation_flights", **ok_args), ts) is None
    # veto: chain broken (LAX->JFK leg cannot follow ATL->PHL) — endpoints from CACHE,
    # even though the args CLAIM legal endpoints (amendment 1e: args endpoints untrusted)
    bad = [dict(new_flights[0]),
           {"flight_number": "HAT020", "date": "2024-05-25", "origin": "PHL", "destination": "ATL"}]
    err = ag._armor_guard(_call(mod, "update_reservation_flights",
                                reservation_id="RES1", cabin="economy",
                                flights=bad, payment_id="credit_card_1"),
                          {**ts, "new_flights": bad})
    assert err is not None and "not available" in err
    # veto: unknown flight number (never seen in any search/reservation JSON)
    ghost = [{"flight_number": "HAT999", "date": "2024-05-21"}, dict(new_flights[1])]
    err = ag._armor_guard(_call(mod, "update_reservation_flights",
                                reservation_id="RES1", cabin="economy",
                                flights=ghost, payment_id="credit_card_1"),
                          {**ts, "new_flights": ghost})
    assert err is not None and "HAT999" in err and "not available" in err
    # veto: args != ts.new_flights (dropped segment)
    err = ag._armor_guard(_call(mod, "update_reservation_flights",
                                reservation_id="RES1", cabin="economy",
                                flights=[dict(new_flights[0])], payment_id="credit_card_1"), ts)
    assert err == mod._ARMOR_ERR_VERBATIM
    # veto: basic economy flight modification (cached cabin, not the 8B's res_cabin)
    res_be = dict(RES1, reservation_id="RESBE", cabin="basic_economy")
    ag._call_args["c_be"] = ("get_reservation_details", {"reservation_id": "RESBE"})
    ag._armor_ingest(_tm(mod, "c_be", json.dumps(res_be)))
    err = ag._armor_guard(_call(mod, "update_reservation_flights",
                                reservation_id="RESBE", cabin="basic_economy",
                                flights=[dict(f) for f in new_flights], payment_id="credit_card_1"),
                          {**ts, "new_flights": new_flights})
    assert err == "Error: basic economy flights cannot be modified"
    # pass: basic economy CABIN-ONLY change (same flight set) is not (f)-vetoed
    same = [{"flight_number": "HAT001", "date": "2024-05-20"},
            {"flight_number": "HAT002", "date": "2024-05-25"}]
    err = ag._armor_guard(_call(mod, "update_reservation_flights",
                                reservation_id="RESBE", cabin="business",
                                flights=same, payment_id="credit_card_1"),
                          {**ts, "new_flights": same})
    assert err is None
    print("PASS test_update_flights_guard")


# --------------------------------------------------------------------------- #
# C) book_reservation pre-guard + baggage rewrite
# --------------------------------------------------------------------------- #

def test_book_guard():
    mod = _load(True)
    ag = _agent(mod)
    _seed_caches(mod, ag)
    ts = {"membership": "silver", "pay_methods": USER1["payment_methods"]}
    base = dict(
        user_id="mia_li_3668", origin="ATL", destination="PHL", flight_type="one_way",
        cabin="economy",
        flights=[{"flight_number": "HAT010", "date": "2024-05-21"}],
        passengers=[{"first_name": "A", "last_name": "B", "dob": "1990-01-01"},
                    {"first_name": "C", "last_name": "D", "dob": "1991-01-01"}],
        payment_methods=[{"payment_id": "certificate_1", "amount": 250},
                         {"payment_id": "credit_card_1", "amount": 50}],
        total_baggages=5, nonfree_baggages=5, insurance="no",
    )
    # pass + rewrite: silver x economy = 2 free bags/pax * 2 pax = 4 -> nonfree = 5-4 = 1
    call = _call(mod, "book_reservation", **base)
    assert ag._armor_guard(call, ts) is None
    assert call.arguments["nonfree_baggages"] == 1
    # veto: two certificates
    err = ag._armor_guard(_call(mod, "book_reservation", **{**base, "payment_methods": [
        {"payment_id": "certificate_1", "amount": 250},
        {"payment_id": "certificate_2", "amount": 50}]}), ts)
    assert err is not None and "certificate" in err
    # veto: certificate amount over stored balance (250)
    err = ag._armor_guard(_call(mod, "book_reservation", **{**base, "payment_methods": [
        {"payment_id": "certificate_1", "amount": 300}]}), ts)
    assert err == "Error: Not enough balance in payment method certificate_1"
    # veto: unknown payment id
    err = ag._armor_guard(_call(mod, "book_reservation", **{**base, "payment_methods": [
        {"payment_id": "gift_card_777", "amount": 300}]}), ts)
    assert err == "Error: Payment method gift_card_777 not found"
    # veto: more than five passengers
    six = [dict(base["passengers"][0]) for _ in range(6)]
    err = ag._armor_guard(_call(mod, "book_reservation", **{**base, "passengers": six}), ts)
    assert err is not None and "five passengers" in err
    # veto: chain does not run origin->destination (booked LAX->JFK leg for an ATL->PHL trip)
    err = ag._armor_guard(_call(mod, "book_reservation", **{**base, "flights": [
        {"flight_number": "HAT020", "date": "2024-05-21"}]}), ts)
    assert err == mod._ARMOR_ERR_CHAIN
    # pass: onestop chain ATL->CLT->PHL from the cached onestop search
    call = _call(mod, "book_reservation", **{**base, "flights": [
        {"flight_number": "HAT030", "date": "2024-05-21"},
        {"flight_number": "HAT031", "date": "2024-05-21"}]})
    assert ag._armor_guard(call, ts) is None
    print("PASS test_book_guard")


# --------------------------------------------------------------------------- #
# D) update_reservation_baggages rewrite/veto
# --------------------------------------------------------------------------- #

def test_bags_guard():
    mod = _load(True)
    ag = _agent(mod)
    _seed_caches(mod, ag)
    # 8B wrote garbage nonfree + stale ts totals: guard recomputes from CACHED res/user.
    ts = {"membership": "gold",  # 8B mis-parse: cached user says silver -> allowance 2*1=2
          "res_total_baggages": 99, "res_nonfree_baggages": 99}
    call = _call(mod, "update_reservation_baggages", reservation_id="RES1",
                 total_baggages=3, nonfree_baggages=9, payment_id="credit_card_1")
    # pass + rewrite: silver x economy x 1 pax = 2 free -> nonfree = 3-2 = 1
    assert ag._armor_guard(call, ts) is None
    assert call.arguments["nonfree_baggages"] == 1
    # veto: reducing totals (cached total is 1)
    err = ag._armor_guard(_call(mod, "update_reservation_baggages", reservation_id="RES1",
                                total_baggages=0, nonfree_baggages=0,
                                payment_id="credit_card_1"), ts)
    assert err == "Error: baggages cannot be removed from a reservation"
    # veto: corrected nonfree below the cached nonfree (only-add rule)
    res2 = dict(RES1, reservation_id="RES2", total_baggages=3, nonfree_baggages=3)
    ag._call_args["c_r2"] = ("get_reservation_details", {"reservation_id": "RES2"})
    ag._armor_ingest(_tm(mod, "c_r2", json.dumps(res2)))
    err = ag._armor_guard(_call(mod, "update_reservation_baggages", reservation_id="RES2",
                                total_baggages=4, nonfree_baggages=4,
                                payment_id="credit_card_1"), ts)
    assert err == "Error: nonfree baggages cannot be reduced on a reservation"
    # fail-open: no cached reservation -> untouched pass-through
    call = _call(mod, "update_reservation_baggages", reservation_id="RESX",
                 total_baggages=2, nonfree_baggages=7, payment_id="credit_card_1")
    assert ag._armor_guard(call, ts) is None and call.arguments["nonfree_baggages"] == 7
    print("PASS test_bags_guard")


# --------------------------------------------------------------------------- #
# E) cancel_reservation recomputed gate (+ epoch tracker)
# --------------------------------------------------------------------------- #

def test_cancel_guard():
    mod = _load(True)
    ag = _agent(mod)
    _seed_caches(mod, ag)
    deny_ts = {"booked_within_24h": True, "any_flown": False,  # 8B lies -> ignored
               "res_cabin": "business", "res_insurance": "yes"}
    # veto: RES1 is old, economy, uninsured — every 8B boolean says yes, facts say no
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RES1"), deny_ts)
    assert err is not None and "not eligible" in err
    # veto (fail closed): reservation never read -> no cached facts -> no cancel
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESX"), {})
    assert err is not None and "has not been verified" in err
    # pass: booked within 24h (created_at recomputed from the cached JSON)
    res24 = dict(RES1, reservation_id="RES24", created_at="2024-05-15T10:00:00")
    ag._call_args["c_24"] = ("get_reservation_details", {"reservation_id": "RES24"})
    ag._armor_ingest(_tm(mod, "c_24", json.dumps(res24)))
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RES24"), {}) is None
    # pass: business cabin
    resb = dict(RES1, reservation_id="RESB", cabin="business")
    ag._call_args["c_b"] = ("get_reservation_details", {"reservation_id": "RESB"})
    ag._armor_ingest(_tm(mod, "c_b", json.dumps(resb)))
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESB"), {}) is None
    # insurance branch, FRESH reason (written this epoch) -> pass
    resi = dict(RES1, reservation_id="RESI", insurance="yes")
    ag._call_args["c_i"] = ("get_reservation_details", {"reservation_id": "RESI"})
    ag._armor_ingest(_tm(mod, "c_i", json.dumps(resi)))
    ag._armor_track_ops([{"op": "set", "path": "task_state.cancel_reason", "value": "health"}],
                        ["cancel_reason"])
    assert ag._reason_epoch == ag._txn_epoch == 0
    ins_ts = {"cancel_reason": "health"}
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESI"), ins_ts) is None
    # amendment 4 fail-closed: ADVANCE sentinel (res_cabin=null) opens a NEW epoch; the
    # RESIDUAL 'health' from the previous transaction no longer counts -> veto
    ag._armor_track_ops([{"op": "set", "path": "task_state.res_cabin", "value": None},
                         {"op": "set", "path": "task_state.txn_action", "value": "cancel"}],
                        ["res_cabin", "txn_action"])
    assert ag._txn_epoch == 1 and ag._reason_epoch == 0
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESI"), ins_ts)
    assert err is not None and "not eligible" in err
    # reason re-written in the SAME patch as the sentinel belongs to the NEW epoch -> pass
    ag._armor_track_ops([{"op": "set", "path": "task_state.res_cabin", "value": None},
                         {"op": "set", "path": "task_state.cancel_reason", "value": "weather"}],
                        ["res_cabin", "cancel_reason"])
    assert ag._txn_epoch == 2 and ag._reason_epoch == 2
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESI"),
                           {"cancel_reason": "weather"}) is None
    # insurance + no_refund_ok -> pass
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESI"),
                           {"no_refund_ok": True}) is None
    # veto: a segment already flew (recomputed from flight dates), even though insured+fresh
    resf = dict(resi, reservation_id="RESF",
                flights=[dict(RES1["flights"][0], date="2024-05-10")])
    ag._call_args["c_f"] = ("get_reservation_details", {"reservation_id": "RESF"})
    ag._armor_ingest(_tm(mod, "c_f", json.dumps(resf)))
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESF"),
                          {"cancel_reason": "weather"})
    assert err is not None and "not eligible" in err
    # pass: airline cancelled a segment, PROVEN via a cached get_flight_status result
    ag._call_args["c_st"] = ("get_flight_status", {"flight_number": "HAT001", "date": "2024-05-20"})
    ag._armor_ingest(_tm(mod, "c_st", "cancelled"))
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RES1"), {}) is None
    # veto: double cancel (cached status already 'cancelled')
    resc = dict(RES1, reservation_id="RESC", status="cancelled")
    ag._call_args["c_c"] = ("get_reservation_details", {"reservation_id": "RESC"})
    ag._armor_ingest(_tm(mod, "c_c", json.dumps(resc)))
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESC"), {})
    assert err is not None and "already cancelled" in err
    print("PASS test_cancel_guard")


# --------------------------------------------------------------------------- #
# veto loop: error feedback re-enters the base loop; exhaustion fails closed
# --------------------------------------------------------------------------- #

def test_veto_loop_and_exhaustion():
    mod = _load(True)
    ag = _agent(mod)  # empty caches -> cancel is always vetoed (fail closed)
    fed = []  # what the base loop was fed on each retry
    vetoable = mod.AssistantMessage(
        role="assistant", content=None,
        tool_calls=[_call(mod, "cancel_reservation", reservation_id="RES1")])

    orig = mod.SchemaSoloAgent.generate_next_message
    def stub(self, message, state):
        fed.append(message)
        return vetoable, state  # the "executor" stubbornly re-emits the same cancel
    mod.SchemaSoloAgent.generate_next_message = stub
    try:
        state = SimpleNamespace(task_state={"resolved": False}, messages=[], finished=False)
        out, _ = mod.SchemaSoloStdAgent.generate_next_message(ag, None, state)
    finally:
        mod.SchemaSoloAgent.generate_next_message = orig
    # 1 initial + _ARMOR_MAX_VETO error-feedback rounds; every feedback is an env-style
    # error ToolMessage answering the vetoed call id; the write NEVER got out: the final
    # message is the fail-closed stop, not the cancel.
    assert len(fed) == 1 + mod._ARMOR_MAX_VETO
    for m in fed[1:]:
        assert isinstance(m, mod.ToolMessage) and m.error and m.content.startswith("Error:")
        assert m.id == "tc1"
    assert out.content == mod.SchemaSoloStdAgent.STOP_TOKEN and state.finished
    print("PASS test_veto_loop_and_exhaustion")


# --------------------------------------------------------------------------- #
# F) deterministic ADVANCE normalizer (smoke-v2 cluster A: tasks 17/18/29)
# --------------------------------------------------------------------------- #

def _ops_dict(ops):
    return {o["path"].split(".")[-1]: o["value"] for o in ops if o.get("op") == "set"}


def test_advance_normalizer():
    mod = _load(True)
    ag = _agent(mod)
    queue = [{"action": "update_baggages", "reservation_id": None,
              "details": "add 3 checked bags", "reason": None}]
    ts = {"txn_action": "update_passengers", "txn_queue": queue,
          "all_res_ids": ["A1", "B2"], "resolved": False}
    # (1) corrupted LLM advance (task-17 line-519 shape: queue=[] instead of the pop,
    #     res_cabin sentinel missing) -> re-derived pop of the PRE-patch queue head
    llm = [{"op": "set", "path": "task_state.write_landed", "value": True},
           {"op": "set", "path": "task_state.txn_action", "value": "update_baggages"},
           {"op": "set", "path": "task_state.txn_queue", "value": []}]
    fixed = _ops_dict(ag._armor_fix_advance("DO_UPDATE_PAX", "doUpdatePax1", list(llm), ts))
    assert fixed["write_landed"] is True                      # non-advance ops survive
    assert fixed["txn_action"] == "update_baggages"
    assert fixed["txn_queue"] is None                          # [] normalized to null
    assert fixed["res_cabin"] is None                          # sentinel always written
    assert fixed["candidate_res_ids"] == ["A1", "B2"]          # restored from all_res_ids
    assert "resolved" not in fixed                             # queue had an entry -> no resolve
    # (2) double ADVANCE from the SAME entry (task-17 line-521 shape: the retest
    #     calculate re-fires the ADVANCE and kills the pending txn) -> suppressed
    llm2 = [{"op": "set", "path": "task_state.txn_action", "value": None},
            {"op": "set", "path": "task_state.txn_queue", "value": None},
            {"op": "set", "path": "task_state.resolved", "value": True}]
    fixed2 = _ops_dict(ag._armor_fix_advance("DO_UPDATE_PAX", "doUpdatePax1", list(llm2),
                                             {"txn_action": "update_baggages", "txn_queue": None}))
    assert fixed2 == {"res_cabin": None}                       # nothing else is touched
    # (3) premature resolved=true while the queue still holds entries (task-18 BOH180
    #     shape: gpt-5.4 dropped the trailing comm txn) -> the pop wins, not the resolve
    ag2 = _agent(mod)
    llm3 = [{"op": "set", "path": "task_state.write_landed", "value": True},
            {"op": "set", "path": "task_state.comm_note", "value": "update X: refunded 5164"},
            {"op": "set", "path": "task_state.txn_action", "value": None},
            {"op": "set", "path": "task_state.resolved", "value": True}]
    q3 = [{"action": "comm", "reservation_id": None, "details": "sum the refunds", "reason": None}]
    fixed3 = _ops_dict(ag2._armor_fix_advance(
        "DO_UPDATE_FLIGHTS", "doUpdateFlights6", list(llm3),
        {"txn_action": "update_flights", "txn_queue": q3, "all_res_ids": ["A1"]}))
    assert fixed3["txn_action"] == "comm" and "resolved" not in fixed3
    assert fixed3["comm_note"] == "update X: refunded 5164"
    # (4) empty queue -> clean terminal resolve; COMM also clears its scratch
    ag3 = _agent(mod)
    fixed4 = _ops_dict(ag3._armor_fix_advance(
        "COMM", "comm1",
        [{"op": "set", "path": "task_state.comm_note", "value": "gift cards total 327"},
         {"op": "set", "path": "task_state.resolved", "value": True}],
        {"txn_action": "comm", "txn_queue": None}))
    assert fixed4["resolved"] is True and fixed4["txn_action"] is None
    assert fixed4["txn_queue"] is None and fixed4["search_scratch"] is None
    assert fixed4["comm_note"] == "gift cards total 327"
    # (5) no advance intent (error turn / scratch-only patch) -> untouched
    plain = [{"op": "set", "path": "task_state.search_scratch", "value": "X cost 1"}]
    assert ag3._armor_fix_advance("COMM", "comm2", list(plain), {"txn_queue": None}) == plain
    # (6) non-ADVANCE subtask (GET_PLAN compiles the queue) -> untouched
    gp = [{"op": "set", "path": "task_state.txn_action", "value": "cancel"},
          {"op": "set", "path": "task_state.txn_queue", "value": queue}]
    assert ag3._armor_fix_advance("GET_PLAN", "getPlan", list(gp), {}) == gp
    print("PASS test_advance_normalizer")


# --------------------------------------------------------------------------- #
# G) CHECK_FSTAT / CHECK_FLOWN exhaustion completion (smoke-v2 task 7 loop)
# --------------------------------------------------------------------------- #

def test_check_walk_completion():
    mod = _load(True)
    ag = _agent(mod)
    seg = {"flight_number": "HAT007", "date": "2024-05-19"}
    # last entry popped as intact, terminal flag forgotten -> flight_cancelled=false
    ops = [{"op": "set", "path": "task_state.check_flights", "value": None}]
    fixed = _ops_dict(ag._armor_fix_check("CHECK_FSTAT", list(ops), {"check_flights": [seg]}))
    assert fixed["flight_cancelled"] is False and fixed["check_flights"] is None
    # stuck state: pre check_flights already null and the patch sets nothing terminal
    fixed = _ops_dict(ag._armor_fix_check("CHECK_FSTAT", [], {"check_flights": None}))
    assert fixed["flight_cancelled"] is False
    # entries remain -> untouched
    ops2 = [{"op": "set", "path": "task_state.check_flights", "value": [seg]}]
    assert ag._armor_fix_check("CHECK_FSTAT", list(ops2), {"check_flights": [seg, seg]}) == ops2
    # a patch that DID set a terminal flag is never touched
    ops3 = [{"op": "set", "path": "task_state.flight_cancelled", "value": True},
            {"op": "set", "path": "task_state.check_flights", "value": None}]
    assert ag._armor_fix_check("CHECK_FSTAT", list(ops3), {"check_flights": [seg]}) == ops3
    # CHECK_FLOWN: exhausted same-day walk -> any_flown=false + refill from res_flights
    rf = [dict(seg), {"flight_number": "HAT174", "date": "2024-05-29"}]
    fixed = _ops_dict(ag._armor_fix_check(
        "CHECK_FLOWN", [{"op": "set", "path": "task_state.check_flights", "value": []}],
        {"check_flights": [seg], "res_flights": rf}))
    assert fixed["any_flown"] is False and fixed["check_flights"] == rf
    print("PASS test_check_walk_completion")


# --------------------------------------------------------------------------- #
# H) LOAD_RES / FIND_RES stale read-target rewrite (smoke-v2 task 39 killer)
# --------------------------------------------------------------------------- #

def test_read_target_rewrite():
    mod = _load(True)
    ag = _agent(mod)
    # LOAD_RES: the 8B re-reads the PREVIOUS (just-cancelled) reservation -> forced
    # back to task_state.reservation_id
    call = _call(mod, "get_reservation_details", reservation_id="8C8K4E")
    state = SimpleNamespace(task_state={"reservation_id": "UDMOP1"},
                            pending={"subtask_type": "LOAD_RES", "subtask": "loadRes2"})
    ag._armor_fix_read_target(call, state)
    assert call.arguments["reservation_id"] == "UDMOP1"
    assert ag._call_args["tc1"] == ("get_reservation_details", {"reservation_id": "UDMOP1"})
    # LOAD_RES with the sentinel 'NONE' -> untouched (denyNotFound owns that state)
    call = _call(mod, "get_reservation_details", reservation_id="8C8K4E")
    state = SimpleNamespace(task_state={"reservation_id": "NONE"},
                            pending={"subtask_type": "LOAD_RES", "subtask": "loadRes2"})
    ag._armor_fix_read_target(call, state)
    assert call.arguments["reservation_id"] == "8C8K4E"
    # FIND_RES: contract is the FIRST candidate id
    call = _call(mod, "get_reservation_details", reservation_id="ZZZZZZ")
    state = SimpleNamespace(task_state={"reservation_id": None,
                                        "candidate_res_ids": ["AAA111", "BBB222"]},
                            pending={"subtask_type": "FIND_RES", "subtask": "findRes1"})
    ag._armor_fix_read_target(call, state)
    assert call.arguments["reservation_id"] == "AAA111"
    # other subtasks (e.g. COMM picks its own reads) -> untouched
    call = _call(mod, "get_reservation_details", reservation_id="CCC333")
    state = SimpleNamespace(task_state={"reservation_id": "UDMOP1"},
                            pending={"subtask_type": "COMM", "subtask": "comm1"})
    ag._armor_fix_read_target(call, state)
    assert call.arguments["reservation_id"] == "CCC333"
    print("PASS test_read_target_rewrite")


# --------------------------------------------------------------------------- #
# I) cancel guard: cached 'landed'/'flying' same-day verdict beats date arithmetic
# --------------------------------------------------------------------------- #

def test_cancel_guard_sameday_flown():
    mod = _load(True)
    ag = _agent(mod)
    _seed_caches(mod, ag)
    # business-cabin reservation with a SAME-DAY segment: date arithmetic says not
    # flown (2024-05-15 >= today), but a cached get_flight_status says it departed
    ress = dict(RES1, reservation_id="RESSD", cabin="business",
                flights=[{"flight_number": "HAT050", "date": "2024-05-15",
                          "origin": "ATL", "destination": "PHL", "price": 100}])
    ag._call_args["c_sd"] = ("get_reservation_details", {"reservation_id": "RESSD"})
    ag._armor_ingest(_tm(mod, "c_sd", json.dumps(ress)))
    assert ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESSD"), {}) is None
    ag._call_args["c_sst"] = ("get_flight_status", {"flight_number": "HAT050", "date": "2024-05-15"})
    ag._armor_ingest(_tm(mod, "c_sst", "landed"))
    err = ag._armor_guard(_call(mod, "cancel_reservation", reservation_id="RESSD"), {})
    assert err is not None and "not eligible" in err
    print("PASS test_cancel_guard_sameday_flown")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_off_is_passthrough()   # must run FIRST (imports with the flag unset)
    test_comm_note()
    test_update_flights_guard()
    test_book_guard()
    test_bags_guard()
    test_cancel_guard()
    test_veto_loop_and_exhaustion()
    test_advance_normalizer()
    test_check_walk_completion()
    test_read_target_rewrite()
    test_cancel_guard_sameday_flown()
    print("\nALL AIRLINE ARMOR TESTS PASSED")
