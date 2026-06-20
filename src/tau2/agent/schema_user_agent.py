"""SchemaUserAgent — non-solo (dialogue) schema-driven agent for tau2.

Same schema-walk control layer as :class:`SchemaSoloAgent`, but for the
interactive (user-simulator) setting. In non-solo telecom the agent only holds
the **account/billing** tools (``tools.py``); all device diagnostics/fixes are
**user** tools (``user_tools.py``). So the agent:

* **agent-side subtask** (its tool is in ``self.tools``): real tool call, parse
  the tool result — exactly like solo.
* **user-side subtask** (device tool, not in ``self.tools``): emit ONE natural
  language instruction asking the customer to do it and report back; parse the
  customer's free-text reply into ``task_state`` with the same StateUpdater.

No ticket is given (the whole point is to elicit), so ``task_state`` starts
empty and the agent first asks for the phone number. Termination is by the
user simulator's ``###STOP###`` once it judges the issue resolved (or
``transfer_to_human_agents`` for escalation); there is no ``done`` tool.

Registered as ``schema_user_agent`` (non-solo, no ``solo_mode`` metadata).
Specs via ``SCHEMAFLEX_SPEC_DIR`` — same artifacts as the solo agent.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
from tau2.agent.schema_agent import (
    _JSON_RETRIES,
    SchemaSoloAgent,
    SchemaSoloAgentState,
    _EXECUTOR_USER,
    _MACRO_INIT_SYSTEM,
    _MACRO_INIT_USER,
    _apply_patch_ops,
    _SanitizedTool,
    _extract_json,
    _field,
    _parse_patch_ops,
    _record_tokens,
    _state_log,
    eval_pred,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import generate

# --------------------------------------------------------------------------- #
# Prompts specific to dialogue mode
# --------------------------------------------------------------------------- #

# Executor system prompt WITHOUT the ticket (non-solo must not leak the answer)
# and without the "you have direct device access" claim (device is the user's).
_EXECUTOR_SYSTEM_USER = """You are the EXECUTOR in a schema-driven telecom support agent talking to a live customer.
You are given ONE subtask, the current machine-tracked task_state, and a RESTRICTED set of account/billing tools you can call directly.
Your only job: call EXACTLY ONE tool that advances the current subtask, following the subtask's rules.
Do NOT try to solve the whole issue; the loop will hand you the next subtask afterwards.
You do NOT have access to the customer's device — device actions are requested from the customer separately."""

# Turn a user-side subtask into a single natural-language instruction.
_INSTRUCT_SYSTEM = """You are a friendly telecom support agent talking to a customer.
The customer must perform a device action themselves (you cannot do it for them).
Write ONE short, SELF-CONTAINED instruction: tell them exactly what to do on their phone and what to report back.
- Be concrete. If the guidance names a specific app, setting, or value (e.g. app_name='messaging', permission='storage', a network mode), state it in plain words ("your Messages app", "the Storage permission").
- Ask the customer to perform EXACTLY ONE action (open one screen, toggle one setting, run one check). If that one screen shows several values, ask them to read them all back — reading one screen is still ONE action. But never bundle two SEPARATE actions/settings in one message — some customers refuse a multi-part request.
- NEVER make the customer guess which app/setting you mean, and never ask them for something you were already told.
One or two sentences. No markdown, no lists, no preamble."""

_INSTRUCT_USER = """Current step: {subtask_name} ({subtask_type})
Goal: {goal}

The single device action to ask for now:
{actions}

Guidance (translate any tool args like app_name/permission/value into plain words):
{rules}

Write the single instruction message to the customer now."""

_PHONE_SYSTEM = """Extract the customer's telephone number from their message.
Return ONLY a JSON object: {"phone_number": "<number>"} — or {"phone_number": null} if none is present.
Do not invent a number."""

# StateUpdater for DIALOGUE: the input is the customer's own words (not a tool's
# structured output), so the parse rules — which were written against the tool's
# literal string format — are demoted to hints, and the model is told to map
# everyday wording onto each field's value space (the allowed paths + enum hints).
_STATE_UPDATER_SYSTEM_USER = """You are the STATE_UPDATER in a telecom support loop.
You asked the customer to do something on their phone, and they replied IN THEIR OWN WORDS.
Read their reply and emit patch operations that update the machine-tracked task_state.

Output a single JSON object and NOTHING else:
{{"patch_ops": [ {{"op": "set", "path": "task_state.<field>", "value": <v>}}, ... ]}}

Hard rules:
- Emit a "set" op ONLY for the paths listed under ALLOWED PATHS. Never write any other path.
- The customer speaks casually ("it's still slow", "now it says 4G", "data roaming is off").
  Map their everyday wording onto the field's value space — do NOT require any exact phrasing.
- When a field lists "(one of: ...)", choose the option that best matches what they describe.
- If the reply genuinely says nothing about a field, omit it (do not guess).
- Values come from the customer's reply, not from assumptions.
- No markdown, no comments, no prose — JSON object only."""

_STATE_UPDATER_USER_DLG = """You asked the customer to perform: {action}
(step: {subtask_name} / {subtask_type})

Field hints (how each value usually shows up):
{parse_rules}

ALLOWED PATHS (you may only 'set' these):
{allowed_paths}

Current task_state:
{task_state}

The customer replied:
{reply}

Return the patch_ops JSON now."""


# Non-answers the model sometimes emits when a field isn't stated; never store these.
_NULLISH = {"", "n/a", "na", "none", "null", "unknown", "unspecified", "not provided", "not stated"}


def _is_nullish(v) -> bool:
    return isinstance(v, str) and v.strip().lower() in _NULLISH


# --------------------------------------------------------------------------- #
# SAGE: structured-uncertainty stop/elicit gate (Suri et al. 2026), applied
# training-free to the dialogue elicitation seam. The belief π over a subtask's
# done_when_all aspects is computed in PURE PYTHON from the schema's enum/bool
# domains (no SLM self-verification — verification is offloaded to the schema,
# per the project direction). EVPI − λ·n_a replaces the hand-tuned "ask 3 times
# then skip" heuristic with a domain-size-derived retry budget: a binary aspect
# (|D|=2, low EVPI) is abandoned faster than a 6-way enum (high EVPI).
#   Defaults are the paper's (λ=0.5, α=0.1, ε=1e-4); tunable via env.
#   SCHEMAFLEX_SAGE=1 turns the gate on; off → original stuck≥3 behavior (A/B).
# --------------------------------------------------------------------------- #
_SAGE_ON = os.environ.get("SCHEMAFLEX_SAGE", "") not in ("", "0", "false", "False")
_SAGE_LAMBDA = float(os.environ.get("SCHEMAFLEX_SAGE_LAMBDA", "0.5"))
_SAGE_ALPHA = float(os.environ.get("SCHEMAFLEX_SAGE_ALPHA", "0.1"))
_SAGE_EPS = float(os.environ.get("SCHEMAFLEX_SAGE_EPS", "1e-4"))
_SAGE_MAX_ASKS = int(os.environ.get("SCHEMAFLEX_SAGE_MAX_ASKS", "6"))  # watchdog backstop
_DWA_FIELD_RE = re.compile(r"task_state\.(\w+)")

# --------------------------------------------------------------------------- #
# Dynamic scheduling = SPARSE big-model supervision. The cheap 8B grinds the
# schema walk; a stronger model (gpt-5.4) is invoked ONLY at the hard junctions
# where the static schema gives up (the agent↔user seam). The supervisor's
# action space is CONSTRAINED by the schema (allowed patch paths, the stuck
# subtask, its done-conditions) — "schema 规范 the scheduler" — so the big model
# adds runtime adaptivity (re-extract / rephrase / escalate) without being able
# to hallucinate actions outside the schema. SAGE's give-up gate becomes the
# cheap trigger for when to spend a supervisor call.
#   SCHEMAFLEX_SUPERVISOR=1 turns it on; sparse (≤ MAX calls/episode) for $.
# --------------------------------------------------------------------------- #
_SUPERVISOR_ON = os.environ.get("SCHEMAFLEX_SUPERVISOR", "") not in ("", "0", "false", "False")
_SUPERVISOR_LLM = os.environ.get("SCHEMAFLEX_SUPERVISOR_LLM", "openai/gpt-5.4")
_SUPERVISOR_MAX = int(os.environ.get("SCHEMAFLEX_SUPERVISOR_MAX", "4"))

# --------------------------------------------------------------------------- #
# REPLAN: dynamic, state-aware subtask RE-RANKER (training-free orchestration).
# The static walk (_active_subtask) follows base_subtask_sequence positionally,
# which forces the agent through unanswerable read-only diagnostics before the
# repair→retest→resolved chain — burning the step budget (max_steps) on the
# hardest families. REPLAN re-orders the eligible candidates by a CAUSAL key:
# "is this subtask on the produces/consumes chain toward `resolved`?". On-chain
# repairs/retests are pulled forward; off-chain pure-read diagnostics are
# down-weighted and abandoned faster — never a state-writing repair (which is
# always kept). All gated by SCHEMAFLEX_REPLAN; off → byte-identical positional
# walk (clean A/B). Independent of and composable with SAGE / SUPERVISOR.
# --------------------------------------------------------------------------- #
_REPLAN_ON = os.environ.get("SCHEMAFLEX_REPLAN", "") not in ("", "0", "false", "False")
# A subtask WRITES backend/device state (a "repair") if its subtask_type carries
# a fix-prefix OR any base_action tool mutates. Classify pure-read diagnostics
# (only get_/check_/run_speed_test/can_send_mms) as skippable; anything else is
# treated as a repair (never skipped) — uncertainty resolves to "do not skip".
_REPAIR_PREFIX_RE = re.compile(r"^(FIX_|RESET_|RESUME_|MAKE_|REQUEST_|REBOOT_)")
_MUT_TOOL_RE = re.compile(
    r"^(toggle_|enable_|disable_|set_|make_|reset_|reseat_|unseat_|lock_|unlock_|"
    r"refuel_|grant_|revoke_|disconnect_|resume_|reboot_|send_payment|pay_)"
)
_READONLY_TOOL_RE = re.compile(r"^(get_|check_|run_speed_test|can_send_mms)")

_SUPERVISOR_SYSTEM = """You are a DYNAMIC SCHEDULER supervising a small-model agent troubleshooting with a live customer. The agent normally follows a FIXED schema sequence of subtasks, but it is now DEADLOCKED — stuck repeating one step. Each subtask is an action PRIMITIVE with a goal and a completion condition. You are NOT bound by the fixed sequence: based on the current machine state, pick the single best next move to break the deadlock.

Choose exactly one decision:
- "schedule": invoke ANY primitive next (by exact name from the PRIMITIVE MENU) — the one the current state actually calls for. This is how you break the deadlock: e.g. if a fix has already been applied (its field is already set in the state) but the verification/retest step never ran, schedule the primitive that runs that check instead of re-applying the fix. Pick a name from the MENU only.
- "extract": the customer ALREADY stated the needed info earlier but the agent failed to record it. Return patch_ops that set the missing field(s) from what the customer ACTUALLY said — ALLOWED PATHS only, respect each enum, never invent a value.
- "escalate": genuinely unresolvable in this dialogue (customer refuses/cannot comply, or needs a human). Give a one-line summary.

Prefer "schedule" when the state shows the current step's fix is already done but a downstream check is missing. Output a single JSON object and NOTHING else:
{"diagnosis": "<one line: why deadlocked>", "decision": "schedule|extract|escalate", "target_subtask": "<exact MENU name, for schedule>", "patch_ops": [{"op":"set","path":"task_state.<field>","value": <v>}], "summary": "<for escalate>"}"""

_SUPERVISOR_USER = """DEADLOCKED ON: {subtask_name} ({subtask_type})
Its goal: {goal}
Unfilled fields blocking ITS completion: {unfilled}
It completes when: {done_when}

PRIMITIVE MENU (you may schedule ANY of these next, regardless of sequence order):
{menu}

ALLOWED PATHS (extract may only 'set' these):
{allowed_paths}

Current task_state:
{task_state}

CONVERSATION so far (Customer = the user, Agent = the agent):
{transcript}

Return the decision JSON now."""


def _clean_phone(v):
    """Return a phone string only if it carries enough digits; else None."""
    if not isinstance(v, (str, int)):
        return None
    s = str(v)
    digits = "".join(c for c in s if c.isdigit())
    return s if len(digits) >= 7 else None


class SchemaUserAgentState(SchemaSoloAgentState):
    """Solo walk state + dialogue bookkeeping."""

    asked_phone: bool = False
    seeded: bool = False  # opening narrative parsed into context fields once
    stuck: dict = {}  # subtask name -> consecutive no-progress (empty-write) replies
    asked: dict = {}  # SAGE aspect history n_a: task_state field -> #times elicited
    supervisor_calls: int = 0  # sparse big-model supervisor invocations this episode
    # what we are waiting on: 'tool' (env result), 'user' (reply to an instruction),
    # 'await_phone' (reply to the phone-number question), or None.
    pending_kind: Optional[str] = None


class SchemaUserAgent(SchemaSoloAgent):
    """Non-solo schema agent (see module docstring)."""

    # -- neutralize the solo-only __init__ steps ------------------------- #
    def add_stop_tool(self) -> None:  # no `done` tool in dialogue mode
        return

    def validate_tools(self) -> None:
        names = {t.name for t in self.tools}
        if self.TRANSFER_TOOL_NAME not in names:
            logger.warning(f"Tool {self.TRANSFER_TOOL_NAME} not found — escalation unavailable.")

    @classmethod
    def check_valid_task(cls, task) -> bool:
        # Dialogue tasks carry a user_scenario; nothing to assert here.
        return True

    # -- conversational system prompt (no ticket) ------------------------- #
    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def get_init_state(self, message_history=None) -> SchemaUserAgentState:
        if message_history is None:
            message_history = []
        family = self._pick_family()
        return SchemaUserAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
            task_state={"resolved": False},  # no ticket → start empty, elicit via dialogue
            family=family,
            subtask_done={},
            entered=[],
            pending=None,
            finished=False,
        )

    # -- agent-side tool set + per-subtask routing ------------------------ #
    @property
    def _agent_tool_names(self) -> set:
        return {t.name for t in self.tools}

    def _next_action(self, subtask, state):
        """Pick the next base_action within a (possibly mixed agent+user) subtask.

        Generic RC1 rule — NOT task-specific: choose the first base_action whose
        ``when`` predicate holds against task_state; if none hold, fall back to the
        LAST action (the post-fix retest, whose ``when`` is prose like 'X just
        succeeded' and so never evaluates true). This walks e.g. FIX_LINE_ROAMING
        get_details(agent) → enable_roaming(agent) → run_speed_test(user) as each
        ``when`` flips, instead of looping on the always-available agent read.
        This agent↔user hand-off point is exactly where dynamic scheduling will
        later plug in.
        """
        bas = self.subtask_spec.get(subtask["subtask_type"], {}).get("base_actions", [])
        if not bas:
            return None
        for ba in bas:
            if eval_pred(ba.get("when", "always"), state.task_state):
                return ba
        return bas[-1]

    def _classify(self, subtask, state):
        """Return (side, chosen_base_action, tool_name).

        side='agent' if the chosen action's tool is one the agent holds (account/
        billing), else 'user' (a device action the customer must perform).
        """
        ba = self._next_action(subtask, state)
        if ba is None:
            return "user", None, subtask["subtask_type"]
        tool = ba.get("tool")
        side = "agent" if tool in self._agent_tool_names else "user"
        return side, ba, tool

    # -- SAGE: structured-uncertainty belief over schema aspects ---------- #
    def _field_dom_size(self, family, field):
        """Domain size |D| of a task_state field, read from the schema.

        enum field → #values; bool field → 2; everything else (ids, free
        strings, lists, numbers) → None, treated as an infinite/continuous
        domain (π=ε). Pure schema lookup — no model call.
        """
        schema = (self.task_spec.get(family, {}) or {}).get("task_state_schema", {}) or {}
        enums = schema.get("enums", {}) or {}
        if field in enums and enums[field]:
            return len(enums[field])
        decl = (schema.get("fields", {}) or {}).get(field, "")
        if isinstance(decl, str) and decl.strip().lower().startswith("bool"):
            return 2
        return None

    def _dwa_fields(self, subtask_type):
        """task_state fields referenced by a subtask's done_when_all (its aspects)."""
        spec = self.subtask_spec.get(subtask_type, {}) or {}
        out = []
        for pred in spec.get("done_when_all", []) or []:
            for m in _DWA_FIELD_RE.findall(str(pred)):
                if m not in out:
                    out.append(m)
        return out

    def _sage_belief(self, family, subtask_type, state):
        """Belief / EVPI / redundancy-cost over a subtask's elicitation aspects.

        Returns ``(max_pi, best_score, gate_open, details)``:
        * ``π_f`` = 1.0 if the field is filled, ``1/|D_f|`` if finite, ε if infinite.
        * ``max_pi`` = ∏ π_f = subtask-completion certainty (one candidate).
        * ``EVPI(f)`` = marginal gain in max_pi from resolving f (π_f→1).
        * ``score(f)`` = EVPI(f) − λ·n_a[f]  (redundancy-penalized).
        * ``gate_open`` = some unfilled aspect is still worth asking
          (best_score ≥ α·max_pi). False with unfilled aspects ⇒ give up.
        """
        fields = self._dwa_fields(subtask_type)
        pis = {}
        for f in fields:
            if state.task_state.get(f) is not None:
                pis[f] = 1.0
            else:
                d = self._field_dom_size(family, f)
                pis[f] = (1.0 / d) if d else _SAGE_EPS
        max_pi = 1.0
        for v in pis.values():
            max_pi *= v
        details, best_score = [], float("-inf")
        for f in fields:
            if pis[f] >= 1.0:
                continue  # filled aspect — nothing to elicit
            others = max_pi / pis[f] if pis[f] > 0 else max_pi
            evpi = others * (1.0 - pis[f])
            na = state.asked.get(f, 0)
            score = evpi - _SAGE_LAMBDA * na
            details.append({"field": f, "pi": round(pis[f], 4), "evpi": round(evpi, 4),
                            "na": na, "score": round(score, 4)})
            best_score = max(best_score, score)
        gate_open = bool(details) and best_score >= _SAGE_ALPHA * max_pi
        return max_pi, best_score, gate_open, details

    def _should_give_up(self, subtask, state, tid):
        """Decide whether to abandon (skip) the active subtask.

        SAGE on: give up on a *user-side elicitation* subtask once no unfilled
        aspect is worth asking (best EVPI−cost < α·max_pi), or a watchdog
        backstop (n_a ≥ MAX_ASKS) trips. Agent-side subtasks execute and never
        give up here. SAGE off: original "consecutive no-progress ≥ 3".
        """
        stype = subtask["subtask_type"]
        # REPLAN fast-abandon: a provably off-chain, read-only diagnostic is not
        # worth even one more ask while on-chain work still pends. Double-guarded
        # inside _replan_offchain so a state-writing repair can NEVER reach here.
        if _REPLAN_ON and self._replan_offchain(state, subtask):
            _state_log({"kind": "replan_skip", "task_id": tid, "subtask": subtask["name"],
                        "reason": "off_chain_diag", "task_state": dict(state.task_state)})
            return True
        if not _SAGE_ON:
            return state.stuck.get(subtask["name"], 0) >= 3
        unfilled = [f for f in self._dwa_fields(stype) if state.task_state.get(f) is None]
        if not unfilled:
            return False  # all aspects filled — let done_when_all close it normally
        if any(state.asked.get(f, 0) >= _SAGE_MAX_ASKS for f in unfilled):
            _state_log({"kind": "sage_giveup", "task_id": tid, "subtask": subtask["name"],
                        "reason": "max_asks", "unfilled": unfilled,
                        "task_state": dict(state.task_state)})
            return True
        kind, _, _ = self._classify(subtask, state)
        if kind != "user":
            return False  # agent-side tool advances state deterministically
        max_pi, best_score, gate_open, details = self._sage_belief(state.family, stype, state)
        if not gate_open:
            _state_log({"kind": "sage_giveup", "task_id": tid, "subtask": subtask["name"],
                        "reason": "evpi_below_threshold", "max_pi": round(max_pi, 5),
                        "best_score": round(best_score, 4), "alpha_max_pi": round(_SAGE_ALPHA * max_pi, 5),
                        "details": details, "task_state": dict(state.task_state)})
            return True
        return False

    # -- REPLAN: causal-chain dynamic subtask re-ranker ------------------ #
    def _active_subtask(self, state):  # type: ignore[override]
        """User-mode subtask selector. OFF → parent positional walk (byte-identical)."""
        if not _REPLAN_ON:
            return super()._active_subtask(state)
        return self._active_subtask_replan(state)

    def _subtask_tools(self, subtask_type):
        spec = self.subtask_spec.get(subtask_type, {}) or {}
        return [str(ba.get("tool", "")) for ba in spec.get("base_actions", []) or []]

    def _is_repair(self, subtask):
        """True if the subtask writes backend/device state. Uncertain → True (never skip)."""
        stype = subtask["subtask_type"]
        if _REPAIR_PREFIX_RE.match(stype):
            return True
        return any(_MUT_TOOL_RE.match(t) for t in self._subtask_tools(stype))

    def _is_readonly(self, subtask):
        """True only if EVERY base_action tool is provably read-only (safe to skip)."""
        tools = self._subtask_tools(subtask["subtask_type"])
        return bool(tools) and all(_READONLY_TOOL_RE.match(t) for t in tools)

    def _pred_fields(self, pred):
        """task_state field names referenced by a predicate string or list."""
        out = []
        for p in (pred if isinstance(pred, list) else [pred]):
            for m in _DWA_FIELD_RE.findall(str(p or "")):
                if m not in out:
                    out.append(m)
        return out

    def _produces(self, subtask_type):
        """task_state fields a subtask can write = allowed 'set' paths ∪ parse_rule targets."""
        spec = self.subtask_spec.get(subtask_type, {}) or {}
        allowed = (spec.get("patch_ops_policy", {}) or {}).get("allowed", []) or []
        fields = {_field(str(a.get("path", ""))) for a in allowed
                  if isinstance(a, dict) and a.get("op") == "set"}
        for r in spec.get("parse_rules", []) or []:
            fields.update(_DWA_FIELD_RE.findall(str(r)))
        fields.discard("")
        return fields

    def _consumes(self, subtask):
        """task_state fields referenced by a subtask's entry `when` + base_action `when`s."""
        flds = set(self._pred_fields(subtask.get("when", "always")))
        spec = self.subtask_spec.get(subtask["subtask_type"], {}) or {}
        for ba in spec.get("base_actions", []) or []:
            flds.update(self._pred_fields(ba.get("when", "always")))
        flds.discard("")
        return flds

    def _causal_index(self, state):
        """Static per-family causal index (goal_fields/produces/is_repair/order). Cached."""
        cache = getattr(self, "_replan_cache", None)
        if cache is None:
            cache = self._replan_cache = {}
        fam = state.family
        if fam not in cache:
            fam_spec = self._family_spec(state)
            seq = fam_spec["base_subtask_sequence"]
            goal = set(self._pred_fields(fam_spec.get("success_when_all", [])))
            goal.add("resolved")  # universal gate; produced only by retest/verify
            cache[fam] = {
                "goal_fields": goal,
                "produces": {st["subtask_type"]: self._produces(st["subtask_type"]) for st in seq},
                "is_repair": {st["subtask_type"]: self._is_repair(st) for st in seq},
                "order": {st["name"]: i for i, st in enumerate(seq)},
            }
        return cache[fam]

    def _atom_live(self, atom, ts):
        """An atom is 'live' if its field is still UNKNOWN (could be satisfied) or,
        if KNOWN, actually holds. Falsified-by-known-field ⇒ dead."""
        f = _field(atom.split("==")[0].split("!=")[0])
        if ts.get(f) is None:
            return True
        return eval_pred(atom, ts)

    def _when_live(self, when, ts):
        """A `when` is live if some OR-group has no atom falsified by a KNOWN field.
        Used to prune branches the current state has already ruled out (e.g. the
        billing/suspension subtree once service_status is known 'connected')."""
        pred = (when or "always").strip()
        if pred in ("", "always"):
            return True
        if pred == "never":
            return False
        for grp in pred.split(" or "):
            if all(self._atom_live(a, ts) for a in grp.split(" and ")):
                return True
        return False

    def _needed_fields(self, state, idx):
        """Fields on the path to the goal: goal ∪ consumes of LIVE pending repairs/
        retests, transitively closed through LIVE producers. Dead branches (a repair
        whose `when` is falsified by known state) contribute nothing."""
        ts = state.task_state
        seq = self._family_spec(state)["base_subtask_sequence"]
        pending = [st for st in seq
                   if not state.subtask_done.get(st["name"])
                   and self._when_live(st.get("when", "always"), ts)]
        goal = idx["goal_fields"]
        needed = set(goal)
        for st in pending:
            t = st["subtask_type"]
            if idx["is_repair"][t] or (idx["produces"][t] & goal):
                needed |= self._consumes(st)
        changed = True
        while changed:
            changed = False
            for st in pending:
                if idx["produces"][st["subtask_type"]] & needed:
                    c = self._consumes(st)
                    if not c <= needed:
                        needed |= c
                        changed = True
        return needed

    def _on_chain(self, idx, needed, subtask):
        """Membership: a candidate is on the causal chain if it is a repair/retest, or
        it produces a field needed by a LIVE pending repair/retest toward the goal."""
        t = subtask["subtask_type"]
        prod = idx["produces"][t]
        return idx["is_repair"][t] or bool(prod & idx["goal_fields"]) or bool(prod & needed)

    def _informative(self, state, idx, needed, subtask):
        """A diagnostic is informative iff it can write a NEEDED field that is still
        UNKNOWN — i.e. running it actually advances the causal chain (not a re-check)."""
        ts = state.task_state
        prod = idx["produces"][subtask["subtask_type"]]
        return any(f in needed and ts.get(f) is None for f in prod)

    def _any_repair_done(self, state, idx):
        """A repair has run this episode (so a pending retest is worth pulling forward
        to confirm/close). Prevents premature 'verify' before any fix is applied."""
        return any(idx["is_repair"].get(st["subtask_type"], False)
                   for st in self._family_spec(state)["base_subtask_sequence"]
                   if state.subtask_done.get(st["name"]))

    def _priority(self, state, idx, needed, subtask):
        """4 tiers (high→low), tie-broken by original positional order (stable, so
        ON==OFF absent a higher tier):
          4  eligible REPAIR — apply the fix first.
          3  retest/verify producing the family GOAL-CORE field, AND a repair has run
             (the A1 close: confirm & set resolved with the right check).
          2  informative on-chain diagnostic, or a post-repair retest that only
             produces `resolved` (not the goal-core field).
          1  off-chain, already-informed, OR a premature retest (no repair done yet)."""
        t = subtask["subtask_type"]
        prod = idx["produces"][t]
        goal_core = idx["goal_fields"] - {"resolved"}  # the family's success field(s)
        if idx["is_repair"][t]:
            tier = 4  # eligible repair (candidate set is already when-filtered)
        elif "resolved" in prod:  # pure retest/verify — the CLOSER (produces resolved).
            # NOTE: key on `resolved`, NOT `prod & goal_fields` — a diagnostic like
            # DIAGNOSE_DATA also produces a goal field (speed_test) but is not a closer.
            if not self._any_repair_done(state, idx):
                tier = 1  # premature: nothing fixed yet to confirm
            elif prod & goal_core:
                tier = 3  # the right closer for THIS family
            else:
                tier = 2  # a retest that can't confirm the family goal
        elif self._informative(state, idx, needed, subtask):
            tier = 2
        else:
            tier = 1
        return (tier, -idx["order"][subtask["name"]])

    def _replan_offchain(self, state, subtask):
        """True iff REPLAN on and this subtask is a provably off-chain, read-only
        diagnostic while on-chain work still pends — the ONLY thing REPLAN skips.
        A state-writing repair can never satisfy this (double-guarded)."""
        if not _REPLAN_ON or self._is_repair(subtask) or not self._is_readonly(subtask):
            return False
        idx = self._causal_index(state)
        needed = self._needed_fields(state, idx)
        if self._on_chain(idx, needed, subtask):
            return False
        seq = self._family_spec(state)["base_subtask_sequence"]
        for st in seq:  # is there on-chain work still pending elsewhere?
            if st["name"] == subtask["name"] or state.subtask_done.get(st["name"]):
                continue
            if self._on_chain(idx, needed, st):
                return True
        return False

    def _active_subtask_replan(self, state):
        """Causal-chain re-ranker (REPLAN on). Sticky-preserving; eligible set identical
        to the parent walk; only the PICK ORDER among eligible candidates changes."""
        # 1) stickiness — identical to parent: keep a started-but-not-done subtask.
        if state.active_subtask and not state.subtask_done.get(state.active_subtask):
            st = self._subtask_by_name(state, state.active_subtask)
            if st is not None:
                return st
        ts = state.task_state
        seq = self._family_spec(state)["base_subtask_sequence"]
        eligible = [st for st in seq
                    if not state.subtask_done.get(st["name"])
                    and eval_pred(st.get("when", "always"), ts)]
        if not eligible:
            state.active_subtask = None
            return None
        idx = self._causal_index(state)
        needed = self._needed_fields(state, idx)
        best = max(eligible, key=lambda st: self._priority(state, idx, needed, st))
        self._log_replan(state, idx, needed, eligible, best)
        state.active_subtask = best["name"]
        self._on_enter(best, state)
        return best

    def _log_replan(self, state, idx, needed, eligible, best):
        cands = [{
            "name": st["name"], "type": st["subtask_type"],
            "tier": self._priority(state, idx, needed, st)[0],
            "repair": idx["is_repair"][st["subtask_type"]],
            "on_chain": self._on_chain(idx, needed, st),
            "pos": idx["order"][st["name"]],
        } for st in eligible]
        _state_log({"kind": "replan", "task_id": getattr(self.task, "id", "?"),
                    "chosen": best["name"], "chosen_type": best["subtask_type"],
                    "goal_fields": sorted(idx["goal_fields"]),
                    "candidates": cands, "task_state": dict(state.task_state)})

    def _sage_focus_hint(self, subtask_type, state) -> str:
        """On a re-ask of an unresolved enum aspect, surface the closed option
        set so the customer's reply becomes a closed classification (much easier
        for a small model to parse than open prose). Empty string otherwise."""
        if not _SAGE_ON:
            return ""
        enums = ((self.task_spec.get(state.family, {}) or {}).get("task_state_schema", {}) or {}).get("enums", {}) or {}
        best_field, best_evpi = None, float("-inf")
        _, _, _, details = self._sage_belief(state.family, subtask_type, state)
        for d in details:
            if d["na"] >= 1 and enums.get(d["field"]) and d["evpi"] > best_evpi:
                best_field, best_evpi = d["field"], d["evpi"]
        if best_field is None:
            return ""
        opts = " / ".join(map(str, enums[best_field]))
        return f"- If they are unsure how to describe it, ask them to pick the closest one: {opts}."

    # -- dynamic scheduling: sparse big-model supervisor ------------------ #
    def _transcript(self, state, limit: int = 24) -> str:
        """Agent-visible dialogue only (Customer/Agent turns) — never the
        user-sim's hidden intent. Last `limit` turns."""
        lines = []
        for m in state.messages[-limit:]:
            if isinstance(m, UserMessage) and m.content:
                lines.append(f"Customer: {m.content.strip()}")
            elif isinstance(m, AssistantMessage) and m.content:
                lines.append(f"Agent: {m.content.strip()}")
        return "\n".join(lines) or "(no dialogue yet)"

    def _primitive_menu(self, state) -> str:
        """The full family primitive library the scheduler may invoke — name,
        open/done status, goal, what it does, and its completion condition.
        This is the schema-constrained action space ('schema 规范')."""
        lines = []
        for st in self._family_spec(state)["base_subtask_sequence"]:
            nm, stype = st["name"], st["subtask_type"]
            spec = self.subtask_spec.get(stype, {}) or {}
            status = "done" if state.subtask_done.get(nm) else "open"
            tools = ", ".join(ba.get("tool", "") for ba in spec.get("base_actions", []) if isinstance(ba, dict))
            dwa = "; ".join(spec.get("done_when_all", []) or [])
            lines.append(f"- {nm} [{status}] does: {tools or '-'} | done_when: {dwa or '-'}")
        return "\n".join(lines)

    def _supervise(self, subtask, state, tid):
        """Call the dynamic scheduler (gpt-5.4) at a deadlock; return one of
        ('schedule', target_subtask) | ('extract', written) | ('escalate', summary) | None.
        Decisions are applied through the SAME schema guardrails as the 8B."""
        import json

        stype = subtask["subtask_type"]
        spec = self.subtask_spec.get(stype, {}) or {}
        allowed = (spec.get("patch_ops_policy", {}) or {}).get("allowed", [])
        fam_enums = (self.task_spec.get(state.family, {}).get("task_state_schema", {}) or {}).get("enums", {})

        def _enum_hint(p):
            vals = fam_enums.get(_field(str(p)))
            return f" (one of: {' | '.join(map(str, vals))})" if vals else ""

        allowed_paths = "\n".join(
            f"- {a['path']}{_enum_hint(a['path'])}"
            for a in allowed if isinstance(a, dict) and a.get("op") == "set"
        ) or "- (none)"
        unfilled = [f for f in self._dwa_fields(stype) if state.task_state.get(f) is None]
        user = _SUPERVISOR_USER.format(
            subtask_name=subtask["name"], subtask_type=stype,
            goal=spec.get("goal_template", ""), unfilled=", ".join(unfilled) or "(none)",
            done_when="; ".join(spec.get("done_when_all", []) or []) or "(none)",
            allowed_paths=allowed_paths,
            menu=self._primitive_menu(state),
            task_state=json.dumps(state.task_state, ensure_ascii=False),
            transcript=self._transcript(state),
        )
        obj = None
        for _ in range(_JSON_RETRIES + 1):
            msg = generate(
                model=_SUPERVISOR_LLM,
                messages=[
                    SystemMessage(role="system", content=_SUPERVISOR_SYSTEM),
                    UserMessage(role="user", content=user),
                ],
                call_name="schema_supervisor",
                temperature=0.0, timeout=60.0,
            )
            _record_tokens(tid, "schema_supervisor", msg)
            obj = _extract_json(msg.content or "")
            if isinstance(obj, dict):
                break
        state.supervisor_calls += 1
        if not isinstance(obj, dict):
            return None
        decision = (obj.get("decision") or "").strip().lower()
        log = {"kind": "supervise", "task_id": tid, "subtask": subtask["name"],
               "decision": decision, "diagnosis": obj.get("diagnosis"),
               "unfilled": unfilled, "call_num": state.supervisor_calls}
        if decision == "schedule":
            target = (obj.get("target_subtask") or "").strip()
            valid = self._subtask_by_name(state, target) is not None
            log["target"] = target
            log["valid_target"] = valid
            _state_log(log)
            return ("schedule", target) if valid else None
        if decision == "extract":
            ops = obj.get("patch_ops") or []
            written = _apply_patch_ops(state.task_state, ops, allowed, fam_enums)
            log["written"] = written
            _state_log(log)
            return ("extract", written)
        if decision == "escalate":
            _state_log(log)
            return ("escalate", (obj.get("summary") or "Unresolved after troubleshooting; escalating to a human agent.").strip())
        _state_log(log)
        return None

    def _escalate(self, summary: str, state, tid):
        """Emit a real transfer_to_human_agents tool call (supervisor decision)."""
        call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=self.TRANSFER_TOOL_NAME,
            arguments={"summary": summary},
        )
        am = AssistantMessage(role="assistant", content=None, tool_calls=[call])
        state.task_state["escalated"] = True
        state.pending = None
        state.pending_kind = None
        _state_log({"kind": "escalate", "task_id": tid, "via": "supervisor",
                    "summary": summary, "task_state": dict(state.task_state)})
        state.messages.append(am)
        return am, state

    # -- executor messages WITHOUT the ticket ----------------------------- #
    def _build_executor_messages(self, subtask, state, last_result):
        import json

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
        return [
            SystemMessage(role="system", content=_EXECUTOR_SYSTEM_USER),
            UserMessage(role="user", content=user),
        ]

    # -- dialogue helpers ------------------------------------------------- #
    def _say(self, text: str, state):
        am = AssistantMessage(role="assistant", content=text)
        state.messages.append(am)
        return am, state

    def _parse_phone(self, text: str, state) -> None:
        msg = generate(
            model=self.llm,
            messages=[
                SystemMessage(role="system", content=_PHONE_SYSTEM),
                UserMessage(role="user", content=text),
            ],
            call_name="schema_macro_init",
            **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_macro_init", msg)
        obj = _extract_json(msg.content or "")
        if isinstance(obj, dict):
            ph = _clean_phone(obj.get("phone_number"))
            if ph is not None:
                state.task_state["phone_number"] = ph

    def _seed_context_from_text(self, text: str, state) -> None:
        """Parse the customer's opening narrative into context fields.

        Non-solo has no ticket, so fields the solo MacroInit read from the ticket
        (phone_number, is_abroad, the stated problem) must come from what the
        customer says. We reuse the family's ``init_rules`` MacroInit prompt with
        the narrative in place of the ticket; only fill fields not already known,
        and never overwrite with null (the customer may not mention everything).
        """
        fam_spec = self.task_spec[state.family]
        schema_fields = (fam_spec.get("task_state_schema", {}) or {}).get("fields", {})
        init_rules = fam_spec.get("init_rules", [])
        fields_block = "\n".join(f"- {k}: {v}" for k, v in schema_fields.items())
        rules_block = "\n".join(f"- {r}" for r in init_rules)
        user = _MACRO_INIT_USER.format(
            fields=fields_block, init_rules=rules_block, ticket=text
        )
        obj = self._ask_json(_MACRO_INIT_SYSTEM, user, call_name="schema_macro_init")
        if isinstance(obj, dict):
            for k in schema_fields:
                if k == "resolved":
                    continue
                v = obj.get(k)
                if v is None or _is_nullish(v):
                    continue
                if k == "phone_number":
                    v = _clean_phone(v)
                    if v is None:
                        continue
                if state.task_state.get(k) in (None, "", False):
                    state.task_state[k] = v

    def _phrase_instruction(self, subtask, chosen, state, focus: str = "") -> str:
        spec = self.subtask_spec[subtask["subtask_type"]]
        if chosen is not None:
            actions = f"- {chosen.get('tool')}: {chosen.get('when', '')}"
        else:
            actions = "\n".join(
                f"- {ba['tool']}: {ba.get('when', '')}" for ba in spec.get("base_actions", [])
            )
        rules = "\n".join(f"- {r}" for r in spec.get("executor_sys_rules", []))
        if focus:  # SAGE: surface the closed option set on a re-ask
            rules = f"{rules}\n{focus}" if rules else focus
        user = _INSTRUCT_USER.format(
            subtask_name=subtask["name"],
            subtask_type=subtask["subtask_type"],
            goal=spec.get("goal_template", ""),
            actions=actions,
            rules=rules,
        )
        msg = generate(
            model=self.llm,
            messages=[
                SystemMessage(role="system", content=_INSTRUCT_SYSTEM),
                UserMessage(role="user", content=user),
            ],
            call_name="schema_executor",
            **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_executor", msg)
        return (msg.content or "").strip() or "Could you try that step and tell me what you see?"

    def _update_from_user_reply(self, subtask_type, subtask_name, action_name, reply_text, state) -> None:
        """StateUpdater for a customer's free-text reply (dialogue variant).

        Same allow-list + enum enforcement as the solo updater, but the prompt
        frames the input as casual customer speech and demotes the tool-format
        parse_rules to hints — so 'It shows 4G' updates network_mode even though
        the rule was written for the literal 'Network Mode Preference: <value>'.
        """
        import json

        spec = self.subtask_spec.get(subtask_type, {})
        allowed = (spec.get("patch_ops_policy", {}) or {}).get("allowed", [])
        if not allowed:
            return
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
        parse_rules = "\n".join(f"- {r}" for r in spec.get("parse_rules", [])) or "- (none)"
        user = _STATE_UPDATER_USER_DLG.format(
            action=action_name,
            subtask_name=subtask_name,
            subtask_type=subtask_type,
            parse_rules=parse_rules,
            allowed_paths=allowed_paths,
            task_state=json.dumps(state.task_state, ensure_ascii=False),
            reply=reply_text,
        )
        messages = [
            SystemMessage(role="system", content=_STATE_UPDATER_SYSTEM_USER),
            UserMessage(role="user", content=user),
        ]
        ops = None
        for _ in range(_JSON_RETRIES + 1):
            msg = generate(
                model=self.llm, messages=messages,
                call_name="schema_state_updater", **self._gen_kwargs(),
            )
            _record_tokens(getattr(self.task, "id", "?"), "schema_state_updater", msg)
            ops = _parse_patch_ops(msg.content or "")
            if ops is not None:
                break
            messages = messages + [
                AssistantMessage(role="assistant", content=msg.content or ""),
                UserMessage(role="user", content='Invalid. Return ONLY {"patch_ops": [...]} on the allowed paths.'),
            ]
        written = _apply_patch_ops(state.task_state, ops or [], allowed, fam_enums)
        _state_log({
            "kind": "update", "task_id": getattr(self.task, "id", "?"),
            "subtask": subtask_name, "subtask_type": subtask_type, "tool": action_name,
            "result": reply_text, "patch_ops": ops, "written": written, "source": "user",
            "task_state": dict(state.task_state),
        })
        return written

    def _mark_done_if_complete(self, pend, state) -> None:
        spec = self.subtask_spec.get(pend["subtask_type"], {})
        if eval_pred(spec.get("done_when_all", []), state.task_state):
            state.subtask_done[pend["subtask"]] = True
            if state.active_subtask == pend["subtask"]:
                state.active_subtask = None

    # -- main loop -------------------------------------------------------- #
    def generate_next_message(self, message, state):  # type: ignore[override]
        tid = getattr(self.task, "id", "?")

        # 1) INGEST the incoming message.
        if isinstance(message, UserMessage):
            state.messages.append(message)
            text = message.content or ""
            if state.pending_kind == "await_phone":
                self._parse_phone(text, state)
                state.pending = None
                state.pending_kind = None
            elif state.pending is not None and state.pending_kind == "user":
                pend = state.pending
                written = self._update_from_user_reply(
                    pend["subtask_type"], pend["subtask"], pend["tool_name"], text, state
                )
                self._mark_done_if_complete(pend, state)
                sk = pend["subtask"]
                if state.subtask_done.get(sk):
                    state.stuck.pop(sk, None)
                elif written:
                    state.stuck[sk] = 0  # made progress; reset
                else:
                    state.stuck[sk] = state.stuck.get(sk, 0) + 1  # refused / vague
                # SAGE aspect history n_a: count this elicitation against every
                # done_when_all aspect the reply left unfilled.
                for f in self._dwa_fields(pend["subtask_type"]):
                    if state.task_state.get(f) is None:
                        state.asked[f] = state.asked.get(f, 0) + 1
                state.pending = None
                state.pending_kind = None
            else:
                # Opening problem description: parse it into context fields once
                # (phone_number, is_abroad, … — what solo got from the ticket).
                if not state.seeded:
                    self._seed_context_from_text(text, state)
                    state.seeded = True
                elif not state.task_state.get("phone_number"):
                    self._parse_phone(text, state)
        elif isinstance(message, (ToolMessage, MultiToolMessage)):
            tool_msgs = (
                list(message.tool_messages)
                if isinstance(message, MultiToolMessage)
                else [message]
            )
            for tm in tool_msgs:
                state.messages.append(tm)
            if state.pending is not None and state.pending_kind == "tool" and tool_msgs:
                pend = state.pending
                match = next(
                    (tm for tm in tool_msgs if tm.id == pend.get("tool_call_id")),
                    tool_msgs[-1],
                )
                self._state_update(
                    pend["subtask_type"], pend["subtask"], pend["tool_name"],
                    match.content or "", state,
                )
                self._mark_done_if_complete(pend, state)
                state.pending = None
                state.pending_kind = None
                # Recovery: a failed customer lookup (wrong / hallucinated phone)
                # leaves customer_id unset and would just retry the same bad number
                # forever. Drop the phone and re-ask the user for it.
                if pend["subtask_type"] == "IDENTIFY_CUSTOMER" and not state.task_state.get("customer_id"):
                    state.task_state["phone_number"] = None
                    state.subtask_done.pop(pend["subtask"], None)
                    state.active_subtask = None

        # 2) BOOTSTRAP: we need the phone number before the schema walk can run.
        if not state.task_state.get("phone_number"):
            state.asked_phone = True
            state.pending_kind = "await_phone"
            return self._say(
                "I can help with that. Could you tell me the phone number on the account?",
                state,
            )

        # 3) RESOLVED → closing message; let the user simulator stop.
        if state.task_state.get("resolved"):
            _state_log({"kind": "stop", "task_id": tid, "reason": "resolved",
                        "task_state": dict(state.task_state)})
            return self._say(
                "Great news — that should be fully resolved now. Is there anything else I can help you with?",
                state,
            )

        # 4) Pick the next subtask; abandon any whose elicitation is no longer
        #    worth pursuing. SAGE on: EVPI−cost < α·max_pi (or n_a backstop);
        #    SAGE off: original consecutive no-progress ≥ 3. Either way bounds
        #    time so a refused/vague customer can't loop us to max_steps.
        subtask = self._active_subtask(state)
        while subtask is not None and self._should_give_up(subtask, state, tid):
            # Dynamic scheduling: this give-up point is exactly where the static
            # schema fails. Spend a SPARSE supervisor (big-model) call before
            # abandoning — it may recover within the schema's allowed actions
            # (re-extract missed info / rephrase) or decide to escalate.
            if (_SUPERVISOR_ON and state.supervisor_calls < _SUPERVISOR_MAX
                    and not self._replan_offchain(state, subtask)):
                act = self._supervise(subtask, state, tid)
                if act is not None and act[0] == "extract":
                    self._mark_done_if_complete(
                        {"subtask": subtask["name"], "subtask_type": subtask["subtask_type"]}, state)
                    for f in self._dwa_fields(subtask["subtask_type"]):
                        state.asked.pop(f, None)
                    state.stuck.pop(subtask["name"], None)
                    subtask = self._active_subtask(state)
                    continue  # re-evaluate; subtask may now be done/advanced
                if act is not None and act[0] == "schedule":
                    # DYNAMIC SCHEDULING: jump to ANY primitive, breaking out of
                    # the static base_subtask_sequence deadlock. Re-open the target
                    # (so an already-"done" primitive can re-run), re-fire on_enter,
                    # reset give-up counters, and let the walk pick it next.
                    target = act[1]
                    state.subtask_done.pop(target, None)
                    if target in state.entered:
                        state.entered.remove(target)
                    state.active_subtask = target
                    target_type = (self._subtask_by_name(state, target) or {}).get("subtask_type", "")
                    for f in self._dwa_fields(subtask["subtask_type"]) + self._dwa_fields(target_type):
                        state.asked.pop(f, None)
                    state.stuck.pop(subtask["name"], None)
                    state.stuck.pop(target, None)
                    _state_log({"kind": "reschedule", "task_id": tid,
                                "from": subtask["name"], "to": target,
                                "task_state": dict(state.task_state)})
                    subtask = self._active_subtask(state)
                    continue
                if act is not None and act[0] == "escalate":
                    return self._escalate(act[1], state, tid)
            # default: abandon the subtask (static skip)
            _state_log({"kind": "stuck_skip", "task_id": tid, "subtask": subtask["name"],
                        "task_state": dict(state.task_state)})
            state.subtask_done[subtask["name"]] = True
            state.active_subtask = None
            state.stuck.pop(subtask["name"], None)
            subtask = self._active_subtask(state)
        if subtask is None:
            _state_log({"kind": "stop", "task_id": tid, "reason": "no_active_subtask",
                        "task_state": dict(state.task_state)})
            return self._say(
                "I've done what I can on this; I'll have a specialist follow up with you. Anything else?",
                state,
            )

        # 5) EXECUTE — route to the next base_action (RC1). agent-side = real tool
        #    call (restricted to that one tool); user-side = instruction text.
        kind, chosen, tool_name = self._classify(subtask, state)
        if kind == "agent":
            subset = [_SanitizedTool(t) for t in self.tools if t.name == tool_name]
            if subset:
                messages = self._build_executor_messages(subtask, state, None)
                am = generate(
                    model=self.llm,
                    tools=subset,
                    messages=messages,
                    tool_choice="required",
                    call_name="schema_executor",
                    **self._gen_kwargs(),
                )
                _record_tokens(tid, "schema_executor", am)
                if am.is_tool_call():
                    call = am.tool_calls[0]
                    state.pending = {
                        "subtask": subtask["name"],
                        "subtask_type": subtask["subtask_type"],
                        "tool_name": call.name,
                        "tool_call_id": call.id,
                    }
                    state.pending_kind = "tool"
                    _state_log({"kind": "exec", "task_id": tid, "subtask": subtask["name"],
                                "subtask_type": subtask["subtask_type"], "chosen_tool": call.name,
                                "args": call.arguments, "side": "agent",
                                "task_state": dict(state.task_state)})
                    state.messages.append(am)
                    return am, state

        # user-side (or agent tool unexpectedly unavailable → ask the customer)
        focus = self._sage_focus_hint(subtask["subtask_type"], state)
        instr = self._phrase_instruction(subtask, chosen, state, focus=focus)
        if _SAGE_ON:
            _max_pi, _bs, _go, _det = self._sage_belief(state.family, subtask["subtask_type"], state)
            _state_log({"kind": "sage_belief", "task_id": tid, "subtask": subtask["name"],
                        "max_pi": round(_max_pi, 5), "details": _det, "focus": bool(focus)})
        state.pending = {
            "subtask": subtask["name"],
            "subtask_type": subtask["subtask_type"],
            "tool_name": tool_name,
        }
        state.pending_kind = "user"
        _state_log({"kind": "exec", "task_id": tid, "subtask": subtask["name"],
                    "subtask_type": subtask["subtask_type"], "chosen_tool": tool_name,
                    "side": "user", "instruction": instr,
                    "task_state": dict(state.task_state)})
        return self._say(instr, state)


# --------------------------------------------------------------------------- #
# Factory (registration lives in tau2.registry)
# --------------------------------------------------------------------------- #


def create_schema_user_agent(tools, domain_policy, **kwargs):
    return SchemaUserAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
