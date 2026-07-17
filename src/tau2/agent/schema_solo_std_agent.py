"""Schema solo agent generalized to *standard* (non-telecom) tau2 domains.

The telecom `schema_solo_agent` (see `schema_agent.py`) assumes the task carries a
`ticket` string and that the agent operates the *user's device*. Standard tau2 domains
(retail / airline / banking_knowledge) instead carry a `user_scenario` (no ticket) and
the agent operates backend tools directly on behalf of the user.

This subclass adapts `SchemaSoloAgent` to those domains WITHOUT modifying the shared
`schema_agent.py`:

1. `check_valid_task` — accept scenario-based tasks (ticket may be None) as long as they
   carry evaluation actions.
2. `__init__` — synthesize a `ticket` string from `user_scenario` so every place the base
   reads `self.task.ticket` (system prompt, MacroInit, Executor) works unchanged.
3. `_build_executor_messages` — swap the telecom-flavored executor system prompt
   ("telecom support agent" / "direct access to the user's device") for a domain-neutral
   one. The MacroInit / StateUpdater system prompts are left as-is: their telecom wording
   is cosmetic (outputs are field-filtered by the schema), so it does not bias behavior.

The schema (task_spec.json + subtask_spec.json under SCHEMAFLEX_SPEC_DIR) supplies all
the domain semantics; this file only removes the telecom-specific scaffolding so a
hand-authored retail/airline schema can run on the same control layer.
"""

from __future__ import annotations

import json
import os

from loguru import logger

from tau2.agent.schema_agent import (
    SchemaSoloAgent,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    _EXECUTOR_USER,
    _STATE_UPDATER_SYSTEM,
    _STATE_UPDATER_USER,
    _JSON_RETRIES,
    _field,
    _parse_patch_ops,
    _apply_patch_ops,
    _record_tokens,
    _state_log,
    generate,
)
from tau2.data_model.message import MultiToolMessage, ToolMessage
from tau2.data_model.tasks import Task

# --------------------------------------------------------------------------- #
# L-HILO for the solo path (high-low collaboration)
# --------------------------------------------------------------------------- #
# Same mechanism as schema_user_agent's HILO, ported to the solo StateUpdater:
# the local 8B does every extraction (workhorse); on a small set of LOAD-BEARING
# extractions (the ones whose mis-parse silently dooms the task — e.g. retail's
# NL-options -> variant item_id mapping in CHECK_PRODUCT) a big model (gpt-5.4)
# re-extracts and OVERRIDES. It corrects PARSING, never DECISIONS (the executor /
# control flow are untouched), which is why it does not derail. Budget-capped per
# episode so the 8B stays the workhorse. All env-gated; OFF => byte-identical.
_HILO_ON = os.environ.get("SCHEMAFLEX_HILO", "") not in ("", "0", "false", "False")
_HILO_LLM = os.environ.get("SCHEMAFLEX_HILO_LLM", "openai/gpt-5.4")
_HILO_BASE = os.environ.get("SCHEMAFLEX_HILO_BASE", "")
_HILO_MAX = int(os.environ.get("SCHEMAFLEX_HILO_MAX", "12"))
# Tier-1 load-bearing subtasks: always big-model extraction. Default = retail's
# semantic-arg extractions (variant mapping, return-item selection, order pick).
_HILO_VERIFY = set(
    (os.environ.get("SCHEMAFLEX_HILO_VERIFY")
     or "PLAN_TXNS,POP_TARGET,PLAN_CHANGES,WALK_VARIANT,RESOLVE_RETURN_ITEMS,LOAD_ORDER").split(",")
)

# L-HIEXEC: high-low EXECUTOR. For DECISION-heavy subtasks (which order, which item,
# which variant, multi-item, conditional) the local 8B cannot assemble correct tool
# args; route the executor tool-call of those subtasks to a big model (gpt-5.4) while
# the schema still restricts WHICH tools it may call and the control flow. The 8B stays
# executor for trivial reads and remains the StateUpdater workhorse. This extends the
# high-low split from parsing (HILO) to decisions — the path parse-only HILO can't reach
# on transactional domains. Env-gated; OFF => 8B executor everywhere (byte-identical).
_HIEXEC_ON = os.environ.get("SCHEMAFLEX_HIEXEC", "") not in ("", "0", "false", "False")
_HIEXEC_LLM = os.environ.get("SCHEMAFLEX_HIEXEC_LLM", "openai/gpt-5.4")
_HIEXEC_SET = set(
    (os.environ.get("SCHEMAFLEX_HIEXEC_SUBTASKS")
     or "LOAD_ORDER,CHECK_PRODUCT,RESOLVE_RETURN_ITEMS,"
        "DO_CANCEL,DO_EXCHANGE,DO_RETURN,DO_MODIFY_ITEMS,"
        "DO_MODIFY_ADDRESS,DO_MODIFY_PAYMENT,DO_MODIFY_USER_ADDRESS").split(",")
)

# --------------------------------------------------------------------------- #
# AIRLINE ARMOR — runner-side deterministic hooks for the airline schema
# --------------------------------------------------------------------------- #
# Implements the 5 runner hooks demanded by airline_spec_out/task_spec.json's
# __runner_notes__ (notes 1..5), following the project meta-discovery: never touch
# the 8B's OBSERVATION, only hard-gate its EXECUTION. Mechanism ported from the
# retail guards ("validate the proposed write, do NOT emit the doomed call, feed a
# correction back, rejected attempts never enter the transcript"
# + llm_std_flex_agent's raw-JSON catalog ingest). In the schema loop the feedback
# channel is a synthesized env-style error ToolMessage routed through the base
# generate_next_message, so the subtask's OWN error parse rules do the recovery
# (e.g. DO_UPDATE_FLIGHTS: new_flights=null -> re-search).
#
#   A. comm_note emission on the resolved branch (notes[1])
#   B. update_reservation_flights State Pre-Guard (notes[2] + amendment 1)
#   C. book_reservation State Pre-Guard + baggage rewrite (notes[3] + amendment 2)
#   D. update_reservation_baggages deterministic rewrite (notes[4] + amendment 3)
#   E. cancel_reservation recomputed-fact gate (notes[5] + amendment 4)
#
# All env-gated by SCHEMAFLEX_AIRLINE_ARMOR (default OFF => byte-identical to the
# unarmored agent: every hook body starts with the flag check and falls through to
# the base behavior).
_ARMOR_ON = os.environ.get("SCHEMAFLEX_AIRLINE_ARMOR", "") not in ("", "0", "false", "False")
# Max veto->feedback->retry rounds per generate_next_message call. On exhaustion we
# fail CLOSED (emit the stop token) — a vetoed write must never reach the env.
_ARMOR_MAX_VETO = int(os.environ.get("SCHEMAFLEX_AIRLINE_ARMOR_MAX_VETO", "3"))

# Tools guarded by the armor (everything else passes through untouched).
_ARMOR_GUARDED = {
    "update_reservation_flights",
    "book_reservation",
    "update_reservation_baggages",
    "cancel_reservation",
}

# Free-baggage allowance per passenger: membership x cabin (amendments 2a/3).
_BAG_ALLOWANCE = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}
# Env clock is 2024-05-15T15:00:00 (tools.py _get_datetime). 24h window / flown cutoffs:
_ARMOR_24H_CUTOFF = "2024-05-14T15:00:00"
_ARMOR_TODAY = "2024-05-15"

# ADVANCE-bearing subtasks: their parse_rules end in the mechanical "pop the head of
# txn_queue into the working registers" contract. The LLM decides WHETHER to advance
# (it emits a txn_action/resolved write); Python re-derives WHAT the advance writes —
# smoke v2 showed both the 8B and gpt-5.4 corrupting the mechanics (premature
# resolved=true with a non-empty queue, txn_queue=[] instead of null, missing
# res_cabin sentinel, double-ADVANCE dropping a queued transaction: tasks 17/18/29).
_ADV_SUBTASKS = {
    "DO_CANCEL", "DO_UPDATE_FLIGHTS", "DO_UPDATE_PAX", "DO_UPDATE_BAGS",
    "DO_BOOK", "DENY_TXN", "COMM",
}
# The advance register set that gets re-derived (LLM-authored values are replaced).
_ADV_FIELDS = (
    "txn_action", "txn_queue", "reservation_id", "txn_details",
    "cancel_reason", "res_cabin", "candidate_res_ids", "resolved",
)

# Chain-violation error. Wording carries "not available" on purpose: DO_UPDATE_FLIGHTS'
# error parse rule keys on "a flight is not available" (-> new_flights=null -> re-search)
# and DO_BOOK's on "a flight is not available or not enough seats" (-> book_flights=null).
_ARMOR_ERR_CHAIN = (
    "Error: flight segments do not preserve the reservation origin/destination; "
    "the requested flight list is not available"
)
_ARMOR_ERR_VERBATIM = (
    "Error: the submitted flights do not match the prepared new_flights; "
    "the requested flight list is not available"
)


def _armor_json(text):
    """Best-effort JSON parse of a tool-result/arguments payload; None on failure."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _armor_truthy(v) -> bool:
    """True only for a real true (the 8B sometimes stores 'true' as a string)."""
    if v is True:
        return True
    return isinstance(v, str) and v.strip().lower() == "true"


def _armor_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none"))


def _armor_fd_seq(flights):
    """[(flight_number, date), ...] for a list of flight dicts (None if malformed)."""
    out = []
    for f in flights if isinstance(flights, list) else []:
        if not isinstance(f, dict):
            return None
        out.append((str(f.get("flight_number")), str(f.get("date"))))
    return out


# Domain-neutral counterpart of schema_agent._EXECUTOR_SYSTEM. No "telecom", no device
# framing — the agent calls exactly one backend tool to advance the current subtask.
_EXECUTOR_SYSTEM_STD = """You are the EXECUTOR in a schema-driven control loop for a customer-support agent.
You are given ONE subtask, the current machine-tracked task_state, and a RESTRICTED set of tools.
Your only job: call EXACTLY ONE tool that advances the current subtask, following the subtask's rules.
Do NOT try to solve the whole request; the loop will hand you the next subtask afterwards.

Request:
{ticket}
"""


class SchemaSoloStdAgent(SchemaSoloAgent):
    """`SchemaSoloAgent` for standard (scenario-based) domains. See module docstring."""

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        # Scenario-based domains: accept as long as the task is gradable. Pure-deny
        # tasks (actions == [], graded on DB-unchanged + nl_assertions, e.g. airline
        # task 0) are valid — the correct walk writes nothing.
        # NOTE: do NOT branch on `task.ticket is None` — __init__ synthesizes ticket
        # from user_scenario BEFORE the base-class assert re-runs this check, so the
        # ticket branch would fall through to the base check (requires actions) and
        # reject pure-deny tasks at construction time.
        ec = task.evaluation_criteria
        if ec is None:
            return False
        return bool(
            (ec.actions or [])
            or (getattr(ec, "nl_assertions", None) or [])
            or (getattr(ec, "communicate_info", None) or [])
        )

    def __init__(self, tools, domain_policy, task, llm, llm_args=None):
        # Synthesize a ticket from the user_scenario so the base machinery (which reads
        # self.task.ticket in the system prompt, MacroInit and Executor) works unchanged.
        if getattr(task, "ticket", None) is None and getattr(task, "user_scenario", None) is not None:
            task.ticket = str(task.user_scenario)
        super().__init__(
            tools=tools, domain_policy=domain_policy, task=task, llm=llm, llm_args=llm_args
        )
        self._hilo_calls = 0  # big-model StateUpdater corrections spent this episode (budget cap)
        self._armor_reset()  # per-episode armor caches (pure bookkeeping; inert when armor OFF)

    # ================= AIRLINE ARMOR (SCHEMAFLEX_AIRLINE_ARMOR) ============ #

    def _armor_reset(self) -> None:
        """(Re)initialize the per-episode caches of RAW env facts + patch-stream trackers."""
        self._res_cache = {}       # reservation_id -> latest raw reservation JSON (env truth)
        self._user_cache = None    # latest raw get_user_details JSON (membership, pay methods)
        self._flight_ep = {}       # flight_number -> (origin, destination), from search/res JSONs
        self._fstat = {}           # (flight_number, date) -> status str from get_flight_status
        self._call_args = {}       # tool_call_id -> (tool_name, arguments) of every emitted call
        self._txn_epoch = 0        # bumped on the ADVANCE sentinel (patch sets res_cabin=null)
        self._reason_epoch = None  # epoch in which cancel_reason was last written non-null
        self._comm_emitted = False
        self._advanced = set()     # subtask entry names that already consumed their one ADVANCE

    def get_init_state(self, message_history=None):
        state = super().get_init_state(message_history)
        self._armor_reset()
        # A cancel_reason present in the MacroInit output belongs to epoch 0 (amendment 4).
        if not _armor_null(state.task_state.get("cancel_reason")):
            self._reason_epoch = 0
        return state

    # -- armor: raw-fact caching (flex/mw-agent _ingest_raw pattern) -------- #

    def _armor_record_call(self, msg) -> None:
        """Remember every emitted tool call's (name, args) so incoming ToolMessages can
        be attributed (ToolMessage carries only the call id)."""
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args = tc.arguments if isinstance(tc.arguments, dict) else _armor_json(tc.arguments)
                self._call_args[tc.id] = (tc.name, args if isinstance(args, dict) else {})

    def _armor_ingest(self, message) -> None:
        """Cache the RAW JSON of env tool results (never the 8B's transcription of them)."""
        if isinstance(message, MultiToolMessage):
            tms = list(message.tool_messages)
        elif isinstance(message, ToolMessage):
            tms = [message]
        else:
            return
        for tm in tms:
            if tm.error or not tm.content:
                continue
            name, args = self._call_args.get(tm.id, (None, {}))
            if name == "get_flight_status":
                status = str(tm.content).strip().strip('"').strip("'")
                fn, dt = (args or {}).get("flight_number"), (args or {}).get("date")
                if fn and dt:
                    self._fstat[(str(fn), str(dt))] = status
                continue
            obj = _armor_json(tm.content)
            if isinstance(obj, dict) and obj.get("reservation_id") and "flights" in obj:
                # get_reservation_details AND every write that returns the reservation
                # (book/update_*/cancel) refresh the "latest known reservation" fact.
                self._res_cache[str(obj["reservation_id"])] = obj
                self._armor_index_flights(obj.get("flights"))
            elif isinstance(obj, dict) and obj.get("user_id") and "payment_methods" in obj:
                self._user_cache = obj
            elif isinstance(obj, list):
                # search_direct_flight: [DirectFlight]; search_onestop_flight: [[leg1, leg2]]
                flat = []
                for it in obj:
                    flat.extend(it if isinstance(it, list) else [it])
                self._armor_index_flights(flat)

    def _armor_index_flights(self, flights) -> None:
        """flight_number -> (origin, destination) endpoint index, from env JSON only
        (amendment 1: args-carried endpoints are never trusted)."""
        for f in flights if isinstance(flights, list) else []:
            if isinstance(f, dict) and f.get("flight_number") and f.get("origin") and f.get("destination"):
                self._flight_ep[str(f["flight_number"])] = (str(f["origin"]), str(f["destination"]))

    def _armor_track_ops(self, ops, written) -> None:
        """Track the StateUpdater patch stream for amendment 4: cancel_reason is trusted
        only if written within the CURRENT transaction epoch. The ADVANCE sentinel
        (a patch setting res_cabin=null) bounds epochs; a reason written in the same
        patch as the sentinel belongs to the NEW epoch (ADVANCE writes the next txn)."""
        wr = set(written or [])
        seen = []
        for op in ops if isinstance(ops, list) else []:
            if not isinstance(op, dict) or str(op.get("op")) != "set":
                continue
            path = _field(str(op.get("path", "")))
            if path in wr:
                seen.append((path, op.get("value")))
        if any(p == "res_cabin" and _armor_null(v) for p, v in seen):
            self._txn_epoch += 1
        for p, v in seen:
            if p == "cancel_reason":
                self._reason_epoch = None if _armor_null(v) else self._txn_epoch

    # -- armor: deterministic ADVANCE / check-walk normalizers -------------- #

    @staticmethod
    def _adv_op(path, value):
        return {"op": "set", "path": f"task_state.{path}", "value": value}

    def _armor_fix_advance(self, subtask_type, subtask_name, ops, ts):
        """Deterministic ADVANCE (queue pop). Trigger: an ADVANCE-bearing subtask's
        patch writes txn_action or resolved (= the LLM decided to advance). Action:
        strip every LLM-authored advance-register op and re-derive the pop from the
        PRE-patch txn_queue in Python. One ADVANCE per subtask entry name — each
        duplicate entry (doCancel1..4, ...) serves exactly one transaction by design,
        so a second advance from the same entry (task-17 double-ADVANCE: the retest
        calculate re-fires the ADVANCE parse rule and silently drops the pending
        transaction) is suppressed down to the res_cabin=null close sentinel.
        Non-advance ops (write_landed, comm_note, book_ref, book_*, search_scratch)
        pass through untouched — judgment stays with the LLM, mechanics do not."""
        if subtask_type not in _ADV_SUBTASKS or not isinstance(ops, list):
            return ops
        intent = any(
            isinstance(o, dict) and str(o.get("op")) == "set"
            and _field(str(o.get("path", ""))) in ("txn_action", "resolved")
            for o in ops
        )
        if not intent:
            return ops
        keep = [
            o for o in ops
            if not (isinstance(o, dict) and str(o.get("op")) == "set"
                    and _field(str(o.get("path", ""))) in _ADV_FIELDS)
        ]
        tid = getattr(self.task, "id", "?")
        if subtask_name in self._advanced:
            fixed = keep + [self._adv_op("res_cabin", None)]
            _state_log({"kind": "armor_advance_fix", "task_id": tid,
                        "subtask": subtask_name, "subtask_type": subtask_type,
                        "mode": "suppress_double_advance", "ops_llm": ops})
            return fixed
        queue = ts.get("txn_queue")
        if not isinstance(queue, list):
            queue = []
        if not queue:
            adv = [self._adv_op("txn_action", None), self._adv_op("txn_queue", None),
                   self._adv_op("res_cabin", None), self._adv_op("resolved", True)]
        else:
            head = queue[0] if isinstance(queue[0], dict) else {}
            rest = queue[1:]
            all_ids = ts.get("all_res_ids")
            adv = [self._adv_op("txn_action", head.get("action")),
                   self._adv_op("reservation_id", head.get("reservation_id")),
                   self._adv_op("txn_details", head.get("details")),
                   self._adv_op("cancel_reason", head.get("reason")),
                   self._adv_op("txn_queue", rest if rest else None),
                   self._adv_op("res_cabin", None),
                   self._adv_op("candidate_res_ids",
                                list(all_ids) if isinstance(all_ids, list) else all_ids)]
        if subtask_type == "COMM":
            adv.append(self._adv_op("search_scratch", None))
        self._advanced.add(subtask_name)
        fixed = keep + adv
        if fixed != ops:
            _state_log({"kind": "armor_advance_fix", "task_id": tid,
                        "subtask": subtask_name, "subtask_type": subtask_type,
                        "mode": "normalized", "ops_llm": ops})
        return fixed

    def _armor_fix_check(self, subtask_type, ops, ts):
        """CHECK_FSTAT / CHECK_FLOWN exhaustion completion. When the parse pops the
        LAST check_flights entry as intact but forgets the terminal flag, done_when_all
        can never hold and the subtask loops to max_steps (smoke v2 task 7 burned ~50
        steps re-probing HAT007). This completes the SAME judgment the parse already
        made (segment intact -> pop): if the patch leaves check_flights null/empty and
        sets neither flight_cancelled nor any_flown, write the fail-closed terminal
        value (CHECK_FSTAT: flight_cancelled=false — a deny, never a cancel enabler;
        CHECK_FLOWN: any_flown=false + refill check_flights=res_flights, per its own
        parse rule 4). A patch that DID set a terminal flag is never touched."""
        if subtask_type not in ("CHECK_FSTAT", "CHECK_FLOWN") or not isinstance(ops, list):
            return ops
        sets = {}
        for o in ops:
            if isinstance(o, dict) and str(o.get("op")) == "set":
                sets[_field(str(o.get("path", "")))] = o.get("value")
        if "flight_cancelled" in sets or "any_flown" in sets:
            return ops
        cf = sets["check_flights"] if "check_flights" in sets else ts.get("check_flights")
        if cf:
            return ops  # entries remain -> keep probing
        if subtask_type == "CHECK_FSTAT":
            extra = [self._adv_op("flight_cancelled", False),
                     self._adv_op("check_flights", None)]
        else:
            extra = [self._adv_op("any_flown", False),
                     self._adv_op("check_flights", ts.get("res_flights"))]
        _state_log({"kind": "armor_check_fix", "task_id": getattr(self.task, "id", "?"),
                    "subtask_type": subtask_type, "extra": extra, "ops_llm": ops})
        return ops + extra

    def _armor_fix_read_target(self, call, state):
        """Deterministic read-target rewrite. LOAD_RES's contract is
        get_reservation_details(task_state.reservation_id) and FIND_RES's is the FIRST
        id of candidate_res_ids — both are literal args templates, yet the 8B copies
        the id from the LAST tool result instead (smoke v2 task 39: after ADVANCE to
        UDMOP1 it re-read the just-cancelled 8C8K4E, loaded its facts as the current
        reservation and burned the whole armor veto budget on it; same class in task
        7). Rewrite the argument to the contract value; the walk's own parse rules
        then see the reservation they were written for."""
        if call.name != "get_reservation_details":
            return
        pend = getattr(state, "pending", None) or {}
        sub = pend.get("subtask_type")
        args = call.arguments if isinstance(call.arguments, dict) else _armor_json(call.arguments)
        if not isinstance(args, dict):
            return
        ts = state.task_state
        want = None
        if sub == "LOAD_RES":
            rid = ts.get("reservation_id")
            if rid and str(rid) != "NONE":
                want = str(rid)
        elif sub == "FIND_RES":
            cands = ts.get("candidate_res_ids")
            if isinstance(cands, list) and cands:
                want = str(cands[0])
        if want is not None and str(args.get("reservation_id")) != want:
            _state_log({"kind": "armor_rewrite", "task_id": getattr(self.task, "id", "?"),
                        "tool": call.name, "field": "reservation_id",
                        "was": args.get("reservation_id"), "now": want,
                        "subtask": pend.get("subtask")})
            args["reservation_id"] = want
            call.arguments = args
            self._call_args[call.id] = (call.name, args)  # keep ingest attribution consistent

    # -- armor: the veto/rewrite guards ------------------------------------ #

    def _armor_guard(self, call, ts):
        """Return None to let the call through (possibly with rewritten args), or an
        env-style 'Error: ...' string to veto it. Guards only prove violations from
        cached env facts; unknown facts fail open (mw-agent pattern) EXCEPT the cancel
        gate, which fails closed (an irreversible write must prove eligibility)."""
        args = call.arguments if isinstance(call.arguments, dict) else _armor_json(call.arguments)
        if not isinstance(args, dict):
            args = {}
        if call.name == "update_reservation_flights":
            return self._guard_update_flights(call, args, ts)
        if call.name == "book_reservation":
            return self._guard_book(call, args, ts)
        if call.name == "update_reservation_baggages":
            return self._guard_bags(call, args, ts)
        if call.name == "cancel_reservation":
            return self._guard_cancel(call, args, ts)
        return None

    def _armor_chain_error(self, flights, origin, destination, trip_type):
        """Hook B(a-c)/C(c): per-segment endpoints resolved ONLY from cached env JSON
        (search results + reservation reads); an unresolvable flight_number is vetoed
        outright (amendment 1e). Chain must be continuous and run origin->destination
        (one_way) / origin->destination->origin (round_trip)."""
        eps = []
        for f in flights:
            fn = str((f or {}).get("flight_number") or "")
            ep = self._flight_ep.get(fn)
            if ep is None:
                # exact env wording (tools.py): matches the "flight is not available" parse rules
                return f"Error: Flight {fn} not available on date {(f or {}).get('date')}"
            eps.append(ep)
        if not eps:
            return _ARMOR_ERR_CHAIN
        if origin and eps[0][0] != origin:
            return _ARMOR_ERR_CHAIN
        for a, b in zip(eps, eps[1:]):
            if b[0] != a[1]:
                return _ARMOR_ERR_CHAIN
        if trip_type == "one_way" and destination and eps[-1][1] != destination:
            return _ARMOR_ERR_CHAIN
        if trip_type == "round_trip":
            if origin and eps[-1][1] != origin:
                return _ARMOR_ERR_CHAIN
            if destination and not any(d == destination for _, d in eps):
                return _ARMOR_ERR_CHAIN
        return None

    def _guard_update_flights(self, call, args, ts):
        """Hook B (notes[2] + amendment 1)."""
        rid = str(args.get("reservation_id") or "")
        res = self._res_cache.get(rid)
        flights = args.get("flights")
        if isinstance(flights, str):
            flights = _armor_json(flights)
        got = _armor_fd_seq(flights)
        if not got:
            return _ARMOR_ERR_VERBATIM
        # (1f) basic economy: flight set must be untouched (cabin-only changes pass).
        if isinstance(res, dict) and res.get("cabin") == "basic_economy":
            if set(got) != set(_armor_fd_seq(res.get("flights")) or []):
                return "Error: basic economy flights cannot be modified"
        # (d) args.flights == ts.new_flights verbatim, compared on (flight_number, date)
        # sequence — endpoints/prices in either copy are not part of the contract.
        nf = ts.get("new_flights")
        if isinstance(nf, str):
            nf = _armor_json(nf)
        want = _armor_fd_seq(nf)
        if not want or got != want:
            return _ARMOR_ERR_VERBATIM
        # (a-c) chaining + reservation endpoints, from cached facts (fallback: ts copies).
        origin = (res or {}).get("origin") or ts.get("res_origin")
        destination = (res or {}).get("destination") or ts.get("res_destination")
        trip = (res or {}).get("flight_type") or ts.get("res_trip_type")
        return self._armor_chain_error(flights, origin, destination, trip)

    def _guard_book(self, call, args, ts):
        """Hook C (notes[3] + amendment 2)."""
        passengers = args.get("passengers")
        if isinstance(passengers, str):
            passengers = _armor_json(passengers)
        # (2b) hard cap
        if isinstance(passengers, list) and len(passengers) > 5:
            return "Error: at most five passengers can be booked on one reservation"
        # payment composition + balances (env checks existence/balance/sum but NEVER counts)
        user_pm = ((self._user_cache or {}).get("payment_methods") or {})
        ts_pm = ts.get("pay_methods") if isinstance(ts.get("pay_methods"), dict) else {}
        pms = args.get("payment_methods")
        if isinstance(pms, str):
            pms = _armor_json(pms)
        counts = {}
        for pm in pms if isinstance(pms, list) else []:
            if not isinstance(pm, dict):
                continue
            pid = str(pm.get("payment_id") or "")
            rec = user_pm.get(pid) if isinstance(user_pm.get(pid), dict) else None
            if rec is None:
                rec = ts_pm.get(pid) if isinstance(ts_pm.get(pid), dict) else None
            if rec is None:
                return f"Error: Payment method {pid} not found"
            src = str(rec.get("source") or "")
            counts[src] = counts.get(src, 0) + 1
            if src in ("certificate", "gift_card"):
                bal, amt = rec.get("amount"), pm.get("amount")
                try:
                    if bal is not None and amt is not None and float(amt) > float(bal):
                        return f"Error: Not enough balance in payment method {pid}"
                except (TypeError, ValueError):
                    pass
        if counts.get("certificate", 0) > 1 or counts.get("credit_card", 0) > 1 \
                or counts.get("gift_card", 0) > 3:
            return ("Error: invalid payment combination: at most one certificate, one "
                    "credit card and three gift cards can be used per reservation")
        # (2c) chain check against the args endpoints the env will write verbatim.
        flights = args.get("flights")
        if isinstance(flights, str):
            flights = _armor_json(flights)
        if _armor_fd_seq(flights):
            err = self._armor_chain_error(
                flights, str(args.get("origin") or ""), str(args.get("destination") or ""),
                str(args.get("flight_type") or ""))
            if err:
                return err
        # (2a) deterministic baggage rewrite — never veto, just fix nonfree_baggages.
        membership = (self._user_cache or {}).get("membership") or ts.get("membership")
        cabin = str(args.get("cabin") or "")
        per_pax = _BAG_ALLOWANCE.get(str(membership) or "", {}).get(cabin)
        if per_pax is not None and isinstance(passengers, list) and passengers:
            try:
                tb = int(args.get("total_baggages") or 0)
            except (TypeError, ValueError):
                tb = 0
            nb = max(0, tb - per_pax * len(passengers))
            if args.get("nonfree_baggages") != nb:
                _state_log({"kind": "armor_rewrite", "task_id": getattr(self.task, "id", "?"),
                            "tool": call.name, "field": "nonfree_baggages",
                            "was": args.get("nonfree_baggages"), "now": nb})
            args["total_baggages"] = tb
            args["nonfree_baggages"] = nb
            call.arguments = args
        return None

    def _guard_bags(self, call, args, ts):
        """Hook D (notes[4] + amendment 3): rewrite from cached env facts; the 8B's
        ts['res_total_baggages'] etc. are logged for contrast but never used as inputs."""
        rid = str(args.get("reservation_id") or "")
        res = self._res_cache.get(rid)
        membership = (self._user_cache or {}).get("membership") or ts.get("membership")
        per_pax = _BAG_ALLOWANCE.get(str(membership) or "", {}).get(str((res or {}).get("cabin") or ""))
        pax = len((res or {}).get("passengers") or [])
        old_tb, old_nb = (res or {}).get("total_baggages"), (res or {}).get("nonfree_baggages")
        try:
            tb = int(args.get("total_baggages"))
        except (TypeError, ValueError):
            tb = None
        if res is None or per_pax is None or not pax or old_tb is None or old_nb is None or tb is None:
            _state_log({"kind": "armor_skip", "task_id": getattr(self.task, "id", "?"),
                        "tool": call.name, "why": "facts_unavailable", "args": args})
            return None  # cannot prove anything -> fail open (mw-agent rule)
        if tb < int(old_tb):
            return "Error: baggages cannot be removed from a reservation"
        nb = max(0, tb - per_pax * pax)
        if nb < int(old_nb):
            return "Error: nonfree baggages cannot be reduced on a reservation"
        if args.get("nonfree_baggages") != nb or ts.get("res_total_baggages") != old_tb:
            _state_log({"kind": "armor_rewrite", "task_id": getattr(self.task, "id", "?"),
                        "tool": call.name, "field": "nonfree_baggages",
                        "was": args.get("nonfree_baggages"), "now": nb,
                        "ts_claims": {"res_total_baggages": ts.get("res_total_baggages"),
                                      "res_nonfree_baggages": ts.get("res_nonfree_baggages"),
                                      "new_nonfree_baggages": ts.get("new_nonfree_baggages")},
                        "cached": {"total": old_tb, "nonfree": old_nb}})
        args["total_baggages"] = tb
        args["nonfree_baggages"] = nb
        call.arguments = args
        return None

    def _guard_cancel(self, call, args, ts):
        """Hook E (notes[5] + amendment 4): recompute EVERY eligibility fact in Python
        from the cached raw reservation JSON; the 8B-set booleans are ignored. Fails
        CLOSED — no cached reservation, no cancel."""
        rid = str(args.get("reservation_id") or "")
        res = self._res_cache.get(rid)
        if not isinstance(res, dict):
            return f"Error: reservation {rid} cannot be cancelled: eligibility has not been verified"
        if res.get("status") == "cancelled":
            return f"Error: reservation {rid} is already cancelled"
        flights = [f for f in (res.get("flights") or []) if isinstance(f, dict)]
        dates = [str(f.get("date")) for f in flights if f.get("date")]
        not_flown = all(d >= _ARMOR_TODAY for d in dates)
        if not_flown:
            # A same-day segment can already have departed: trust a cached
            # get_flight_status verdict ('landed'/'flying') over the date arithmetic.
            for f in flights:
                st = self._fstat.get((str(f.get("flight_number")), str(f.get("date"))))
                if st in ("landed", "flying"):
                    not_flown = False
                    break
        within24h = str(res.get("created_at") or "") >= _ARMOR_24H_CUTOFF
        business = res.get("cabin") == "business"
        insured = res.get("insurance") == "yes"
        # amendment 4: cancel_reason counts only if (re)written during THIS txn epoch;
        # a value left over from a previous reservation's cycle is treated as 'other'.
        reason_fresh = self._reason_epoch is not None and self._reason_epoch == self._txn_epoch
        reason = str(ts.get("cancel_reason") or "").strip().lower()
        reason_ok = insured and reason_fresh and reason in ("health", "weather")
        no_refund_ok = insured and _armor_truthy(ts.get("no_refund_ok"))
        flight_cancelled = any(
            self._fstat.get((str(f.get("flight_number")), str(f.get("date")))) == "cancelled"
            for f in flights
        )
        allowed = flight_cancelled or (
            not_flown and (within24h or business or reason_ok or no_refund_ok)
        )
        _state_log({"kind": "armor_cancel_facts", "task_id": getattr(self.task, "id", "?"),
                    "rid": rid, "within24h": within24h, "not_flown": not_flown,
                    "business": business, "insured": insured, "reason": reason,
                    "reason_fresh": reason_fresh, "no_refund_ok": no_refund_ok,
                    "flight_cancelled": flight_cancelled, "allowed": allowed,
                    "ts_claims": {k: ts.get(k) for k in
                                  ("booked_within_24h", "any_flown", "res_cabin",
                                   "res_insurance", "cancel_reason", "flight_cancelled")}})
        if allowed:
            return None
        return (f"Error: reservation {rid} cannot be cancelled: it is not eligible for "
                f"cancellation under the airline policy")

    # -- armor: hook A (comm_note) + the guarded main loop ------------------ #

    def _emit_done(self, state):
        """Hook A (notes[1]): when the walk resolves with a pending comm_note, emit ONE
        final message carrying the note AND the stop token. Rationale for deviating from
        the note's two-step plan (text message first, stop next step): in solo mode the
        orchestrator terminates on any non-stop text message with AGENT_ERROR
        (orchestrator.py l.877-881) and the evaluator zeroes premature terminations
        (evaluator.py l.113); meanwhile stop detection is a SUBSTRING test, not verbatim
        (llm_agent.py is_stop: `STOP_TOKEN in content`), so a combined message terminates
        as a clean AGENT_STOP and the CommunicateEvaluator scans its text content."""
        if _ARMOR_ON and not self._comm_emitted:
            note = state.task_state.get("comm_note")
            if not _armor_truthy(state.task_state.get("resolved")):
                note = None  # only the resolved branch reports (notes[1])
            if not _armor_null(note):
                self._comm_emitted = True
                msg = super()._emit_done(state)
                msg.content = f"{note}\n{self.STOP_TOKEN}"
                _state_log({"kind": "armor_comm", "task_id": getattr(self.task, "id", "?"),
                            "comm_note": str(note)})
                return msg
        return super()._emit_done(state)

    def generate_next_message(self, message, state):  # type: ignore[override]
        if not _ARMOR_ON:
            return super().generate_next_message(message, state)
        self._armor_ingest(message)  # cache raw env facts BEFORE the walk consumes them
        msg, state = super().generate_next_message(message, state)
        self._armor_record_call(msg)
        for _ in range(_ARMOR_MAX_VETO):
            if not (isinstance(msg, AssistantMessage) and msg.is_tool_call()):
                return msg, state
            call = msg.tool_calls[0]
            self._armor_fix_read_target(call, state)  # LOAD_RES/FIND_RES stale-id rewrite
            if call.name not in _ARMOR_GUARDED:
                return msg, state
            err = self._armor_guard(call, state.task_state)
            if err is None:
                return msg, state
            # Veto: the doomed call never reaches the env/transcript. Feed an env-style
            # error ToolMessage back through the base loop so the pending subtask's OWN
            # error parse rules recover (retail mw-agent's reject-and-refeed pattern).
            _state_log({"kind": "armor_veto", "task_id": getattr(self.task, "id", "?"),
                        "tool": call.name, "error": err, "args": call.arguments})
            err_tm = ToolMessage(id=call.id, role="tool", content=err,
                                 requestor="assistant", error=True)
            msg, state = super().generate_next_message(err_tm, state)
            self._armor_record_call(msg)
        # Retries exhausted: fail closed. Ending the episode (stop token) is always
        # DB-safe; emitting the vetoed write never is.
        _state_log({"kind": "armor_veto_exhausted", "task_id": getattr(self.task, "id", "?"),
                    "task_state": dict(state.task_state)})
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            return self._emit_done(state), state
        return msg, state

    def _build_executor_messages(self, subtask, state, last_result):
        spec = self.subtask_spec[subtask["subtask_type"]]
        rules = "\n".join(f"- {r}" for r in spec.get("executor_sys_rules", []))
        actions = "\n".join(
            f"- {ba['tool']}: {ba.get('when', '')}" for ba in spec["base_actions"]
        )
        last = f"Last tool result:\n{last_result}\n\n" if last_result else ""
        user = _EXECUTOR_USER.format(
            subtask_name=subtask["name"],
            subtask_type=subtask["subtask_type"],
            goal=spec.get("goal_template", ""),
            rules=rules,
            actions=actions,
            task_state=json.dumps(state.task_state, ensure_ascii=False),
            last_result=last,
        )
        system = _EXECUTOR_SYSTEM_STD.format(ticket=self.task.ticket or "")
        return [
            SystemMessage(role="system", content=system),
            UserMessage(role="user", content=user),
        ]

    # -- L-HILO StateUpdater override ------------------------------------- #

    def _executor_model(self, subtask) -> str:
        if _HIEXEC_ON and subtask.get("subtask_type") in _HIEXEC_SET:
            return _HIEXEC_LLM
        return super()._executor_model(subtask)

    def _executor_gen_kwargs(self, subtask) -> dict:
        if _HIEXEC_ON and subtask.get("subtask_type") in _HIEXEC_SET:
            return self._hi_gen_kwargs()
        return super()._executor_gen_kwargs(subtask)

    def _hi_gen_kwargs(self) -> dict:
        """Gen kwargs for the big-model corrector: temp 0, proxy base (.env OPENAI_API_BASE
        unless SCHEMAFLEX_HILO_BASE overrides). NOT the local-vLLM api_base."""
        kw = {"temperature": 0.0}
        if _HILO_BASE:
            kw["api_base"] = _HILO_BASE
        return kw

    def _run_updater_once(self, messages, model, gen_kwargs, call_name, tid):
        """One StateUpdater extraction with JSON-retry. Returns parsed patch_ops or None."""
        ops, msgs = None, messages
        for _ in range(_JSON_RETRIES + 1):
            msg = generate(model=model, messages=msgs, call_name=call_name, **gen_kwargs)
            _record_tokens(tid, call_name, msg)
            ops = _parse_patch_ops(msg.content or "")
            if ops is not None:
                break
            msgs = msgs + [
                AssistantMessage(role="assistant", content=msg.content or ""),
                UserMessage(role="user", content='Invalid. Return ONLY {"patch_ops": [...]} on the allowed paths.'),
            ]
        return ops

    def _state_update(self, subtask_type, subtask_name, tool_name, result_text, state) -> None:
        """Solo StateUpdater + L-HILO. 8B extracts every result (workhorse); on a Tier-1
        load-bearing subtask (subtask_type in _HILO_VERIFY) a big model re-extracts and
        overrides — sparse, budget-capped. Corrects parsing only; never the executor.
        OFF (SCHEMAFLEX_HILO unset) => identical to the base solo updater."""
        spec = self.subtask_spec.get(subtask_type, {})
        allowed = (spec.get("patch_ops_policy", {}) or {}).get("allowed", [])
        if not allowed:
            return  # e.g. ESCALATE — nothing to track

        fam_enums = (
            (self.task_spec.get(state.family, {}).get("task_state_schema", {}) or {}).get("enums", {})
        )

        def _enum_hint(path):
            vals = fam_enums.get(_field(str(path)))
            return f" (one of: {' | '.join(map(str, vals))})" if vals else ""

        allowed_paths = "\n".join(
            f"- {a['path']}{_enum_hint(a['path'])}"
            for a in allowed if isinstance(a, dict) and a.get("op") == "set"
        )
        parse_rules = "\n".join(f"- {r}" for r in spec.get("parse_rules", []))
        user = _STATE_UPDATER_USER.format(
            subtask_name=subtask_name,
            subtask_type=subtask_type,
            parse_rules=parse_rules,
            allowed_paths=allowed_paths,
            sys_output_format=spec.get("sys_output_format", '{"patch_ops": []}'),
            task_state=json.dumps(state.task_state, ensure_ascii=False),
            tool_name=tool_name,
            result=result_text,
        )
        messages = [
            SystemMessage(role="system", content=_STATE_UPDATER_SYSTEM),
            UserMessage(role="user", content=user),
        ]
        tid = getattr(self.task, "id", "?")
        ops = self._run_updater_once(messages, self.llm, self._gen_kwargs(), "schema_state_updater", tid)

        if _HILO_ON and self._hilo_calls < _HILO_MAX and subtask_type in _HILO_VERIFY:
            ops_hi = self._run_updater_once(
                messages, _HILO_LLM, self._hi_gen_kwargs(), "schema_state_updater_hi", tid)
            if ops_hi is not None:
                self._hilo_calls += 1
                _state_log({"kind": "hilo", "task_id": tid, "subtask_type": subtask_type,
                            "tool": tool_name, "ops_8b": ops, "ops_hi": ops_hi,
                            "n_hilo": self._hilo_calls})
                ops = ops_hi

        if ops is None:
            logger.warning(f"[schema_state_updater] {subtask_type}: no valid patch_ops; task_state unchanged.")
            _state_log({
                "kind": "update", "task_id": tid, "subtask": subtask_name,
                "subtask_type": subtask_type, "tool": tool_name, "result": result_text,
                "patch_ops": None, "written": [], "task_state": dict(state.task_state),
            })
            return
        if _ARMOR_ON:
            # Deterministic queue mechanics: normalize the ADVANCE pop and complete an
            # exhausted CHECK_FSTAT/CHECK_FLOWN walk — WHETHER to advance / how a status
            # was judged stays the LLM's call, WHAT the registers become does not.
            ops = self._armor_fix_advance(subtask_type, subtask_name, ops, state.task_state)
            ops = self._armor_fix_check(subtask_type, ops, state.task_state)
        written = _apply_patch_ops(state.task_state, ops, allowed, fam_enums)
        if _ARMOR_ON:
            self._armor_track_ops(ops, written)  # amendment 4: cancel_reason freshness
        _state_log({
            "kind": "update", "task_id": tid, "subtask": subtask_name,
            "subtask_type": subtask_type, "tool": tool_name, "result": result_text,
            "patch_ops": ops, "written": written, "task_state": dict(state.task_state),
        })


def create_schema_solo_std_agent(tools, domain_policy, **kwargs):
    return SchemaSoloStdAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
