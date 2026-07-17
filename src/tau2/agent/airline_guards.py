"""Deterministic airline State Pre-Guards for USER-MODE agents (llm_agent_std_flex).

Judgment logic is a self-contained port of the four AIRLINE ARMOR guards in
schema_solo_std_agent.py (hooks B/C/D/E). Deliberately NOT imported from there:
that module drags in the whole schema stack (schema_agent, task_spec machinery)
which user-mode runs must not depend on. The guard rules are copied and adapted.

Meta-discovery compliance: never touch the 8B's observation; only hard-gate its
EXECUTION on facts cached from raw env ToolMessages (never the 8B's transcription).

Differences vs the schema-solo armor (user mode has no task_state / patch stream):
  * cancel_reason: the solo armor reads ts['cancel_reason'] + txn-epoch freshness.
    User mode has neither -> the reason is derived from the USER's own messages via
    conservative health/weather regexes, FAIL-CLOSED: no clear mention => treated
    as 'other' (cancel allowed only via within-24h / business / verified
    flight-cancelled). ts['no_refund_ok'] has no user-mode counterpart -> dropped.
  * update_reservation_flights: the solo (d) check "args.flights == ts.new_flights
    verbatim" is schema-walk-specific -> dropped. BE lock + segment-chain kept.
  * chain check: solo vetoes an endpoint-unresolvable flight_number outright
    (its walk can recover via error parse rules). User mode FAILS OPEN on unknown
    endpoints (cannot prove a violation; the env still rejects truly-bad flights).
  * book payment checks: solo falls back to ts['pay_methods']; user mode has no
    ts -> if get_user_details was never observed, payment checks FAIL OPEN.
All four guards otherwise keep the solo semantics (incl. cancel failing CLOSED:
an irreversible write must prove eligibility from a cached reservation).
"""

from __future__ import annotations

import json
import re

# Airline write tools covered by the guards. Everything else passes through.
GUARDED_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "send_certificate",
}

# Free-baggage allowance per passenger: membership x cabin (airline policy).
BAG_ALLOWANCE = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}
# Env clock is frozen at 2024-05-15T15:00:00 (domains/airline/tools.py _get_datetime).
TODAY = "2024-05-15"
CUTOFF_24H = "2024-05-14T15:00:00"

# cancel_reason from the user's OWN words, fail-closed (see module docstring).
HEALTH_RE = re.compile(
    r"\b(health|sick|ill(ness)?|injur\w*|medical|hospital|surgery|doctor)\b", re.I)
WEATHER_RE = re.compile(
    r"\b(weather|storm|hurricane|typhoon|blizzard|snowstorm|flood(ing)?)\b", re.I)


def _json(x):
    """Best-effort JSON parse; passes dict/list through; None on failure."""
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None


def _fd_seq(flights):
    """[(flight_number, date), ...] for a list of flight dicts (None if malformed)."""
    out = []
    for f in flights if isinstance(flights, list) else []:
        if not isinstance(f, dict):
            return None
        out.append((str(f.get("flight_number")), str(f.get("date"))))
    return out


class AirlineGuardCache:
    """Per-episode cache of RAW env facts + the four write guards.

    ingest() must be fed every successful ToolMessage (name, args, raw content).
    guard() takes a proposed ToolCall; returns None to let it through (possibly
    with args rewritten in place, e.g. nonfree_baggages), or a human-readable
    reason string to veto it. It never mutates args when it vetoes.
    """

    def __init__(self):
        self.res_cache = {}    # reservation_id -> latest raw reservation JSON
        self.user_cache = None  # latest raw get_user_details JSON
        self.flight_ep = {}    # flight_number -> (origin, destination), env JSON only
        self.fstat = {}        # (flight_number, date) -> status str from get_flight_status

    # -- fact ingestion ----------------------------------------------------- #

    def ingest(self, tool_name, args, content):
        if not content:
            return
        if tool_name == "get_flight_status":
            a = args if isinstance(args, dict) else {}
            fn, dt = a.get("flight_number"), a.get("date")
            if fn and dt:
                self.fstat[(str(fn), str(dt))] = str(content).strip().strip('"').strip("'")
            return
        obj = _json(content)
        if isinstance(obj, dict) and obj.get("reservation_id") and "flights" in obj:
            # get_reservation_details AND every write that returns the reservation
            # (book/update_*/cancel) refresh the "latest known reservation" fact.
            self.res_cache[str(obj["reservation_id"])] = obj
            self._index_flights(obj.get("flights"))
        elif isinstance(obj, dict) and obj.get("user_id") and "payment_methods" in obj:
            self.user_cache = obj
        elif isinstance(obj, list):
            # search_direct_flight: [flight]; search_onestop_flight: [[leg1, leg2]]
            flat = []
            for it in obj:
                flat.extend(it if isinstance(it, list) else [it])
            self._index_flights(flat)

    def _index_flights(self, flights):
        for f in flights if isinstance(flights, list) else []:
            if isinstance(f, dict) and f.get("flight_number") and f.get("origin") \
                    and f.get("destination"):
                self.flight_ep[str(f["flight_number"])] = (str(f["origin"]),
                                                           str(f["destination"]))

    # -- guard dispatch ------------------------------------------------------ #

    def guard(self, call, user_text=""):
        """None => pass (args possibly rewritten in place); str => veto reason."""
        args = call.arguments if isinstance(call.arguments, dict) else _json(call.arguments)
        if not isinstance(args, dict):
            args = {}
        if call.name == "cancel_reservation":
            return self._guard_cancel(args, user_text)
        if call.name == "update_reservation_flights":
            return self._guard_update_flights(args)
        if call.name == "book_reservation":
            return self._guard_book(call, args)
        if call.name == "update_reservation_baggages":
            return self._guard_bags(call, args)
        if call.name == "send_certificate":
            # Airline policy grants no self-serve compensation: across the 50-task
            # ground truth send_certificate is never a correct write, while the 8B
            # occasionally volunteers one under user pressure (smoke: task 2
            # regression). An unmatched write always breaks the binary DB check,
            # so a blanket veto can only help.
            return ("compensation certificates cannot be issued for this request "
                    "under airline policy")
        return None

    # -- segment chain (shared by update_flights / book) --------------------- #

    def _chain_error(self, flights, origin, destination, trip_type):
        """Continuity + endpoint check with per-segment endpoints resolved ONLY from
        cached env JSON. Unknown flight_number => fail OPEN (user-mode difference)."""
        eps = []
        for f in flights:
            fn = str((f or {}).get("flight_number") or "")
            ep = self.flight_ep.get(fn)
            if ep is None:
                return None  # cannot prove a violation -> let the env decide
            eps.append(ep)
        if not eps:
            return None
        bad = ("the flight segments do not form a valid continuous "
               "itinerary for this reservation's origin/destination")
        if origin and eps[0][0] != origin:
            return bad
        for a, b in zip(eps, eps[1:]):
            if b[0] != a[1]:
                return bad
        if trip_type == "one_way" and destination and eps[-1][1] != destination:
            return bad
        if trip_type == "round_trip":
            if origin and eps[-1][1] != origin:
                return bad
            if destination and not any(d == destination for _, d in eps):
                return bad
        return None

    # -- update_reservation_flights (BE lock + chain) ------------------------ #

    def _guard_update_flights(self, args):
        rid = str(args.get("reservation_id") or "")
        res = self.res_cache.get(rid)
        flights = args.get("flights")
        if isinstance(flights, str):
            flights = _json(flights)
        got = _fd_seq(flights)
        if not got:
            return "the flights list is empty or malformed — this would wipe the reservation's flights"
        # BE modification lock: flight set must be untouched (payment/cabin-only passes).
        if isinstance(res, dict) and res.get("cabin") == "basic_economy":
            if set(got) != set(_fd_seq(res.get("flights")) or []):
                return "basic economy reservations cannot have their flights modified"
        origin = (res or {}).get("origin")
        destination = (res or {}).get("destination")
        trip = (res or {}).get("flight_type")
        return self._chain_error(flights, origin, destination, trip)

    # -- book_reservation (payment combo + chain + baggage rewrite) ---------- #

    def _guard_book(self, call, args):
        passengers = args.get("passengers")
        if isinstance(passengers, str):
            passengers = _json(passengers)
        if isinstance(passengers, list) and len(passengers) > 5:
            return "at most five passengers can be booked on one reservation"
        # payment composition + balances, from the cached user profile (env truth).
        # No cached profile => fail open (user-mode difference: no ts fallback).
        user_pm = (self.user_cache or {}).get("payment_methods") or {}
        pms = args.get("payment_methods")
        if isinstance(pms, str):
            pms = _json(pms)
        if user_pm:
            counts = {}
            for pm in pms if isinstance(pms, list) else []:
                if not isinstance(pm, dict):
                    continue
                pid = str(pm.get("payment_id") or "")
                rec = user_pm.get(pid) if isinstance(user_pm.get(pid), dict) else None
                if rec is None:
                    return f"payment method {pid} is not in the user's profile"
                src = str(rec.get("source") or "")
                counts[src] = counts.get(src, 0) + 1
                if src in ("certificate", "gift_card"):
                    bal, amt = rec.get("amount"), pm.get("amount")
                    try:
                        if bal is not None and amt is not None and float(amt) > float(bal):
                            return f"payment method {pid} does not have enough balance"
                    except (TypeError, ValueError):
                        pass
            if counts.get("certificate", 0) > 1 or counts.get("credit_card", 0) > 1 \
                    or counts.get("gift_card", 0) > 3:
                return ("invalid payment combination: at most one certificate, one "
                        "credit card and three gift cards can be used per reservation")
        # chain check against the args endpoints the env will write verbatim.
        flights = args.get("flights")
        if isinstance(flights, str):
            flights = _json(flights)
        if _fd_seq(flights):
            err = self._chain_error(
                flights, str(args.get("origin") or ""), str(args.get("destination") or ""),
                str(args.get("flight_type") or ""))
            if err:
                return err
        # deterministic baggage rewrite — never veto, just fix nonfree_baggages.
        membership = (self.user_cache or {}).get("membership")
        per_pax = BAG_ALLOWANCE.get(str(membership or ""), {}).get(str(args.get("cabin") or ""))
        if per_pax is not None and isinstance(passengers, list) and passengers:
            try:
                tb = int(args.get("total_baggages") or 0)
            except (TypeError, ValueError):
                tb = 0
            nb = max(0, tb - per_pax * len(passengers))
            args["total_baggages"] = tb
            args["nonfree_baggages"] = nb
            call.arguments = args
        return None

    # -- update_reservation_baggages (only-increase + rewrite) --------------- #

    def _guard_bags(self, call, args):
        rid = str(args.get("reservation_id") or "")
        res = self.res_cache.get(rid)
        membership = (self.user_cache or {}).get("membership")
        per_pax = BAG_ALLOWANCE.get(str(membership or ""), {}).get(
            str((res or {}).get("cabin") or ""))
        pax = len((res or {}).get("passengers") or [])
        old_tb, old_nb = (res or {}).get("total_baggages"), (res or {}).get("nonfree_baggages")
        try:
            tb = int(args.get("total_baggages"))
        except (TypeError, ValueError):
            tb = None
        if res is None or per_pax is None or not pax or old_tb is None \
                or old_nb is None or tb is None:
            return None  # cannot prove anything -> fail open
        if tb < int(old_tb):
            return "baggages cannot be removed from a reservation"
        nb = max(0, tb - per_pax * pax)
        if nb < int(old_nb):
            return "the number of paid (non-free) baggages cannot be reduced"
        args["total_baggages"] = tb
        args["nonfree_baggages"] = nb
        call.arguments = args
        return None

    # -- cancel_reservation (recomputed eligibility, FAIL-CLOSED) ------------ #

    def _guard_cancel(self, args, user_text):
        rid = str(args.get("reservation_id") or "")
        res = self.res_cache.get(rid)
        if not isinstance(res, dict):
            return (f"reservation {rid} cannot be cancelled yet: its details have not "
                    f"been looked up, so eligibility is unverified (call "
                    f"get_reservation_details first)")
        if res.get("status") == "cancelled":
            return f"reservation {rid} is already cancelled"
        flights = [f for f in (res.get("flights") or []) if isinstance(f, dict)]
        dates = [str(f.get("date")) for f in flights if f.get("date")]
        not_flown = all(d >= TODAY for d in dates)
        if not_flown:
            # a same-day segment may already have departed: trust a cached
            # get_flight_status verdict ('landed'/'flying') over date arithmetic.
            for f in flights:
                st = self.fstat.get((str(f.get("flight_number")), str(f.get("date"))))
                if st in ("landed", "flying"):
                    not_flown = False
                    break
        within24h = str(res.get("created_at") or "") >= CUTOFF_24H
        business = res.get("cabin") == "business"
        insured = res.get("insurance") == "yes"
        # user-mode cancel_reason: only the USER's own clear health/weather mention
        # counts; anything else is 'other' (fail-closed; see module docstring).
        reason_ok = insured and bool(
            HEALTH_RE.search(user_text or "") or WEATHER_RE.search(user_text or ""))
        flight_cancelled = any(
            self.fstat.get((str(f.get("flight_number")), str(f.get("date")))) == "cancelled"
            for f in flights
        )
        allowed = flight_cancelled or (
            not_flown and (within24h or business or reason_ok)
        )
        if allowed:
            return None
        return (f"reservation {rid} is not eligible for cancellation under airline "
                f"policy (not booked within the last 24h, not business cabin, no "
                f"insurance-covered health/weather reason stated by the customer, and "
                f"no airline-cancelled segment; if the customer claims a flight was "
                f"cancelled by the airline, verify it with get_flight_status first)")
