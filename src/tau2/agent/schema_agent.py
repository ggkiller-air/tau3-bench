"""SchemaSoloAgent — a schema-driven solo agent for tau2.

A first-class tau2 agent (registered as ``schema_solo_agent`` in ``tau2.registry``,
alongside ``llm_agent_solo``). It subclasses the reference ``LLMSoloAgent`` and runs
the LightWM/ESSA loop with the small model in three roles:

* **MacroStateInitializer** (one model call per episode): read the ticket + the
  family's ``init_rules`` + ``task_state_schema`` and emit the initial ``task_state``.
* **Executor** (one model call per step): pick ONE tool call that advances the
  active subtask, restricted to that subtask's allowed tools (native tool-calling).
* **StateUpdater** (one model call per tool result): read the tool result + the
  subtask's ``parse_rules`` / ``patch_ops_policy`` / ``sys_output_format`` and emit
  ``patch_ops`` over ``task_state``. Python validates each op against the
  per-subtask allow-list and applies it. NO regex, NO task-specific parsing.

Python keeps only the **generic, schema-driven control layer**: it walks the
family's ``base_subtask_sequence`` and evaluates each subtask's ``when`` /
``done_when_all`` predicate (a tiny DSL interpreter, not task logic — ESSA
evaluates ``done_when_all`` in Python too). It also applies the schema's
declarative ``on_enter_reset`` (invalidate stale fields on subtask entry) and
``retest_guard`` (once a fix has landed, restrict the executor to the retest).

Schema artifact = ``memory/{task_spec.json, subtask_spec.json}`` at the repo root
(hand-authored; auto-generation is the later generalization goal; override the dir
with ``SCHEMAFLEX_SPEC_DIR``). See the SchemaFlex notes in ``CLAUDE.md``.

Solo runs use the stock ``--user dummy_user`` (the earlier ``schema_dummy_user``
workaround is gone now that ``DummyUser.__init__`` tolerates build_user's kwargs).
"""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgentState, LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import generate

# How many times to re-ask the model when it returns unparseable JSON.
_JSON_RETRIES = 2

# --------------------------------------------------------------------------- #
# Schema loading
# --------------------------------------------------------------------------- #

# memory/ specs live at the repo root (tau2-bench/memory), NOT inside the package.
# From src/tau2/agent/schema_agent.py: agent -> tau2 -> src -> tau2-bench == parents[3].
# Override with SCHEMAFLEX_SPEC_DIR.
_DEFAULT_SPEC_DIR = Path(__file__).resolve().parents[3] / "memory"


def _strip_meta(obj):
    """Recursively drop ``__*__`` annotation keys so specs are clean to walk."""
    if isinstance(obj, dict):
        return {k: _strip_meta(v) for k, v in obj.items() if not (k.startswith("__") and k.endswith("__"))}
    if isinstance(obj, list):
        return [_strip_meta(v) for v in obj]
    return obj


def _load_specs():
    spec_dir = Path(os.environ.get("SCHEMAFLEX_SPEC_DIR", _DEFAULT_SPEC_DIR))
    task_spec = _strip_meta(json.loads((spec_dir / "task_spec.json").read_text()))
    subtask_spec = _strip_meta(json.loads((spec_dir / "subtask_spec.json").read_text()))
    return task_spec, subtask_spec


# Map a task-id prefix "[<prefix>]" to a task-family key in task_spec.json.
# The three families exercised by telecom tasks_small.json, one per workflow path:
#   mobile_data_issue -> mobile_data (Path 2: 2.1 unavailable + 2.2 slow)
#   service_issue     -> no_service  (Path 1: no service / connection)
#   mms_issue         -> mms         (Path 3: picture / group messaging)
_FAMILY_BY_PREFIX = {
    "mobile_data_issue": "mobile_data",
    "service_issue": "no_service",
    "mms_issue": "mms",
}

# --------------------------------------------------------------------------- #
# when / done_when_all evaluator  (safe; NOT eval())  — generic schema DSL
# --------------------------------------------------------------------------- #


def _literal(tok: str):
    tok = tok.strip()
    if tok == "true":
        return True
    if tok == "false":
        return False
    if tok == "null":
        return None
    if len(tok) >= 2 and tok[0] in ("'", '"') and tok[-1] == tok[0]:
        return tok[1:-1]  # strip single OR double quotes (specs mix both styles)
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _field(lhs: str) -> str:
    """Strip a leading ``task_state.`` so both predicates and patch paths normalize."""
    lhs = lhs.strip()
    return lhs[len("task_state."):] if lhs.startswith("task_state.") else lhs


def _eval_atom(atom: str, ts: dict) -> bool:
    atom = atom.strip()
    for op in ("==", "!="):
        if op in atom:
            lhs, rhs = atom.split(op, 1)
            lval = ts.get(_field(lhs))
            rval = _literal(rhs)
            # The 8B intermittently stores booleans/nulls as quoted strings
            # ("true"/"false"/"null"), so a `== true` guard or done_when_all atom
            # would silently never match. Normalize the stored side when comparing
            # against a non-string literal (generic; not task-specific).
            if isinstance(lval, str) and not isinstance(rval, str):
                _low = lval.strip().lower()
                if _low == "true":
                    lval = True
                elif _low == "false":
                    lval = False
                elif _low in ("null", "none"):
                    lval = None
            return (lval == rval) if op == "==" else (lval != rval)
    # bare field => truthiness
    return bool(ts.get(_field(atom)))


def eval_pred(pred, ts: dict) -> bool:
    """Evaluate a ``when`` string or a ``done_when_all`` list against task_state.

    Grammar (matches the authored specs): atoms ``task_state.<f> ==|!= <lit>``,
    joined by ``and`` / ``or``, no parentheses. A list is implicit-AND.
    """
    if isinstance(pred, list):
        return all(eval_pred(p, ts) for p in pred)
    pred = (pred or "").strip()
    if pred in ("", "always"):
        return True
    if pred == "never":
        return False
    for or_group in pred.split(" or "):
        if all(_eval_atom(a, ts) for a in or_group.split(" and ")):
            return True
    return False


# --------------------------------------------------------------------------- #
# JSON extraction + schema-validated patch_ops application (control layer)
# --------------------------------------------------------------------------- #


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced ``{...}`` object out of a model response."""
    if not isinstance(text, str):
        return None
    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_patch_ops(text: str) -> Optional[list]:
    """Parse ``{"patch_ops": [...]}`` (or a bare op) out of a model response."""
    obj = _extract_json(text)
    if obj is None:
        return None
    ops = obj.get("patch_ops")
    if ops is None and all(k in obj for k in ("op", "path", "value")):
        ops = [obj]  # tolerate a single bare op
    if isinstance(ops, dict):
        ops = [ops]
    return ops if isinstance(ops, list) else None


def _apply_patch_ops(ts: dict, ops: list, allowed, enums=None) -> list:
    """Apply ``set`` ops to task_state, keeping ONLY paths in the allow-list.

    Returns the list of paths actually written (for logging). Anything off the
    allow-list, malformed, or non-``set`` is dropped silently — the schema's
    ``patch_ops_policy.allowed`` is the contract.

    ``enums`` (field -> allowed-values list, from ``task_state_schema.enums``)
    adds a value-level guard for categorical fields: a value that is not None and
    not in the field's declared set is dropped. This is a schema-level prior, not
    task-specific Python — it stops the 8B from writing a label fragment as a value
    (e.g. ``service_status='Cellular Connection: no_service'``) and nudges it to
    pick a real option (e.g. ``sim_status='locked_pin'``).
    """
    enums = enums or {}
    allowed_pairs = {
        ("set", _field(str(a.get("path", ""))))
        for a in (allowed or [])
        if isinstance(a, dict) and str(a.get("op")) == "set"
    }
    written = []
    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict) or str(op.get("op")) != "set" or "value" not in op:
            continue
        path = _field(str(op.get("path", "")))
        if ("set", path) not in allowed_pairs:
            continue
        val = op["value"]
        if isinstance(val, str):  # the 8B often quotes booleans/nulls -> store real types
            _low = val.strip().lower()
            if _low == "true":
                val = True
            elif _low == "false":
                val = False
            elif _low in ("null", "none"):
                val = None
        allowed_vals = enums.get(path)
        if allowed_vals is not None and val is not None and val not in allowed_vals:
            logger.debug(f"[patch_ops] dropped off-enum {path}={val!r} (allowed: {allowed_vals})")
            continue
        ts[path] = val
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Tool-schema sanitization for the executor
# --------------------------------------------------------------------------- #
# Pydantic emits `anyOf` + `$ref`/`$defs` for params typed `Union[Enum, str]`
# (e.g. telecom's set_network_mode_preference: `mode: Union[NetworkModePreference,
# str]`). vLLM's qwen tool-call parser chokes on that and returns an EMPTY
# arguments string, which litellm then rejects ("Invalid JSON: EOF ... ''") ->
# the whole task dies as an infra error. We send the model a flattened schema
# instead (inline $defs, collapse anyOf of scalar/enum branches into one type).
# This only changes what the MODEL sees; the env still executes the real tool by
# name with its real signature. Generic — also helps other domains' union params.


def _flatten_schema_node(node, defs):
    if isinstance(node, list):
        return [_flatten_schema_node(v, defs) for v in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        name = str(node["$ref"]).split("/")[-1]
        return _flatten_schema_node(dict(defs.get(name, {})), defs)
    if "anyOf" in node:
        branches = [_flatten_schema_node(b, defs) for b in node["anyOf"]]
        # Prefer the most informative scalar branch: an enum if present, else the
        # first concrete branch. Carry a description if the union node had one.
        chosen = next((b for b in branches if isinstance(b, dict) and "enum" in b), None) or branches[0]
        out = dict(chosen)
        if "description" in node and "description" not in out:
            out["description"] = node["description"]
        return {k: v for k, v in out.items() if k != "title"}
    return {k: _flatten_schema_node(v, defs) for k, v in node.items() if k not in ("$defs", "title")}


def _sanitize_openai_schema(schema: dict) -> dict:
    """Inline $defs/$ref and flatten anyOf so vLLM tool-callers can fill args."""
    sch = copy.deepcopy(schema)
    fn = sch.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
        params = fn["parameters"]
        defs = params.get("$defs", {})
        fn["parameters"] = _flatten_schema_node(
            {k: v for k, v in params.items() if k != "$defs"}, defs
        )
    return sch


class _SanitizedTool:
    """Duck-typed Tool view exposing a sanitized ``openai_schema`` to ``generate``.

    ``generate()`` only reads ``tool.name`` / ``tool.openai_schema``; execution
    of the chosen call happens in the environment against the real tool, so a
    schema-only view is safe.
    """

    def __init__(self, tool):
        self._tool = tool
        self.name = tool.name

    @property
    def openai_schema(self):
        return _sanitize_openai_schema(self._tool.openai_schema)


# --------------------------------------------------------------------------- #
# Per-role token accounting (macro / executor / updater)
# --------------------------------------------------------------------------- #
# The StateUpdater and MacroInit are side calls that never land in the sim's
# `messages`, so their tokens are invisible in results.json. We accumulate ALL
# three roles' usage here, keyed by task id, and (if SCHEMAFLEX_TOKEN_LOG is set)
# flush to a sidecar JSON next to results.json so the report can show true cost.
_SCHEMAFLEX_TOKENS: dict = {}
_ROLE_BY_CALL = {
    "schema_macro_init": "macro",
    "schema_executor": "executor",
    "schema_state_updater": "updater",
}


def _record_tokens(task_id: str, call_name: str, msg, sim_seed=None) -> None:
    role = _ROLE_BY_CALL.get(call_name)
    if role is None:
        return
    u = getattr(msg, "usage", None)
    if isinstance(u, dict):
        p, c = (u.get("prompt_tokens") or 0), (u.get("completion_tokens") or 0)
    elif u is not None:
        p, c = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
    else:
        return
    # Key per-SIM = (task_id, seed) so num-trials>1 trials stay SEPARABLE — pass^4 decay
    # analysis needs per-trial token/context, joinable back to results.json by (task_id,
    # seed). seed is distinct per trial (set via agent.set_seed); None → falls back to
    # task_id (fine for single-trial runs).
    key = f"{task_id}␟{sim_seed}" if sim_seed is not None else task_id
    rec = _SCHEMAFLEX_TOKENS.setdefault(key, {
        "task_id": task_id, "seed": sim_seed,
        "macro": 0, "executor": 0, "updater": 0, "prompt": 0, "completion": 0, "calls": 0,
        "peak_prompt": 0, "peak_prompt_exec": 0,
    })
    rec[role] += p + c          # per-role total (prompt+completion)
    rec["prompt"] += p          # run-level prompt total (the cacheable bulk)
    rec["completion"] += c
    rec["calls"] += 1
    # Effective-context tracking: prompt_tokens IS the resident KV / context length of that
    # call. peak_prompt = max over all calls; peak_prompt_exec = max over EXECUTOR calls =
    # peak CONVERSATION context (the decision calls that grow with the dialogue — the prime
    # suspect for long-episode 8B degradation → pass^4 decay).
    if p > rec["peak_prompt"]:
        rec["peak_prompt"] = p
    if role == "executor" and p > rec["peak_prompt_exec"]:
        rec["peak_prompt_exec"] = p
    path = os.environ.get("SCHEMAFLEX_TOKEN_LOG")
    if path:
        try:
            json.dump(_SCHEMAFLEX_TOKENS, open(path, "w"))
        except Exception:
            pass
    # Per-call JSONL trace (append, O(1)): the prompt-length SERIES per sim, for plotting
    # context growth over the episode. Enabled only if SCHEMAFLEX_TOKEN_TRACE is set.
    tpath = os.environ.get("SCHEMAFLEX_TOKEN_TRACE")
    if tpath:
        try:
            with open(tpath, "a") as f:
                f.write(json.dumps({"task_id": task_id, "seed": sim_seed, "role": role,
                                    "p": p, "c": c, "i": rec["calls"]}) + "\n")
        except Exception:
            pass


def _state_log(rec: dict) -> None:
    """Append one JSONL record to SCHEMAFLEX_STATE_LOG (debug only; no-op if unset).

    Used to make the otherwise-invisible StateUpdater/Executor side-calls auditable:
    each record carries the subtask, the tool + its result text (the ground truth the
    updater must transcribe), the emitted patch_ops, which were actually written, and
    the resulting task_state — so a mis-read step can be spotted by diffing the result
    text against what landed in task_state.
    """
    path = os.environ.get("SCHEMAFLEX_STATE_LOG")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Agent state
# --------------------------------------------------------------------------- #


class SchemaSoloAgentState(LLMAgentState):
    """LLMAgentState + schema-walk bookkeeping."""

    task_state: dict = {}
    family: str = ""
    subtask_done: dict = {}
    active_subtask: Optional[str] = None
    entered: list = []  # subtask names already entered (so on_enter_reset fires once)
    pending: Optional[dict] = None
    finished: bool = False
    regrounded: bool = False  # re-grounding probe fired once before giving up (② )


# --------------------------------------------------------------------------- #
# Prompts — Executor, StateUpdater, MacroInit  (all run on self.llm)
# --------------------------------------------------------------------------- #

_EXECUTOR_SYSTEM = """You are the EXECUTOR in a schema-driven control loop for a telecom support agent.
You are given ONE subtask, the current machine-tracked task_state, and a RESTRICTED set of tools.
Your only job: call EXACTLY ONE tool that advances the current subtask, following the subtask's rules.
Do NOT try to solve the whole ticket; the loop will hand you the next subtask afterwards.
You have direct access to the user's device (solo mode): device actions are yours to perform.

Ticket:
{ticket}
"""

_EXECUTOR_USER = """Current subtask: {subtask_name} ({subtask_type})
Goal: {goal}

Rules for this subtask:
{rules}

Tool options for this subtask (call exactly one):
{actions}

Current task_state (machine-tracked):
{task_state}

{last_result}Call exactly one tool now to advance this subtask."""

_STATE_UPDATER_SYSTEM = """You are the STATE_UPDATER in a schema-driven telecom support loop.
A tool was just called and returned a result. Read the result and emit patch operations that
update the machine-tracked task_state, following this subtask's parse rules EXACTLY.

Output a single JSON object and NOTHING else:
{{"patch_ops": [ {{"op": "set", "path": "task_state.<field>", "value": <v>}}, ... ]}}

Hard rules:
- Emit a "set" op ONLY for the paths listed under ALLOWED PATHS. Never write any other path.
- If the result gives no new value for a field, omit that field (do not guess, do not fabricate).
- Values must come from the tool result, not from the ticket.
- No markdown, no comments, no prose — JSON object only."""

_STATE_UPDATER_USER = """Subtask: {subtask_name} ({subtask_type})

Parse rules:
{parse_rules}

ALLOWED PATHS (you may only 'set' these):
{allowed_paths}

Output format example:
{sys_output_format}

Current task_state:
{task_state}

The tool `{tool_name}` returned:
{result}

Return the patch_ops JSON now."""

_MACRO_INIT_SYSTEM = """You are the INITIALIZER in a schema-driven telecom support loop.
Read the support ticket and produce the INITIAL machine-tracked task_state, following the
init rules EXACTLY.

Output a single JSON object and NOTHING else — the task_state with the listed fields.
Hard rules:
- Use ONLY the fields listed below; do not invent fields.
- Diagnostic fields (speed_test, roaming/data toggles, usage) are unknown at start: use null.
  Never infer them from the ticket text.
- Use null for any value the ticket does not state.
- No markdown, no comments, no prose — JSON object only."""

_MACRO_INIT_USER = """task_state fields (name: type / meaning):
{fields}

Init rules:
{init_rules}

Ticket:
{ticket}

Return the initial task_state as one JSON object now."""


class SchemaSoloAgent(LLMSoloAgent):
    """Solo agent that executes a hand-authored schema (see module docstring)."""

    def __init__(self, tools, domain_policy, task, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, task=task, llm=llm, llm_args=llm_args
        )
        self.task_spec, self.subtask_spec = _load_specs()

    def _gen_kwargs(self) -> dict:
        """self.llm_args, plus the extra_body that disables Qwen3 'thinking'.

        Qwen3 emits a long ``<think>...`` block by default, which (a) burns the
        token budget — the executor/StateUpdater calls then hit ``finish_reason=
        length`` and return truncated JSON — and (b) is pure latency for our
        JSON-only / single-tool-call calls. We pass the vLLM chat-template switch
        ``enable_thinking=False`` to suppress it.

        Default ON only for Qwen models (model string contains 'qwen'), so non-Qwen
        endpoints (e.g. the gpt-5.4 proxy) never receive an unknown param. Force with
        ``SCHEMAFLEX_DISABLE_THINKING=1`` / disable with ``=0``.
        """
        kw = dict(self.llm_args or {})
        flag = os.environ.get("SCHEMAFLEX_DISABLE_THINKING")
        if flag is not None:
            disable = flag.strip().lower() in ("1", "true", "yes", "on")
        else:
            disable = "qwen" in (self.llm or "").lower()
        if disable:
            eb = dict(kw.get("extra_body") or {})
            ctk = dict(eb.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            eb["chat_template_kwargs"] = ctk
            kw["extra_body"] = eb
        return kw

    # -- small-model helpers ---------------------------------------------- #

    def _ask_json(self, system: str, user: str, call_name: str) -> Optional[dict]:
        """Run self.llm on a (system,user) pair and return the parsed JSON object.

        Re-asks up to ``_JSON_RETRIES`` times with a corrective nudge. No regex
        fallback — if the model never returns valid JSON we return None and the
        caller decides (StateUpdater: apply nothing; MacroInit: minimal state).
        """
        messages = [
            SystemMessage(role="system", content=system),
            UserMessage(role="user", content=user),
        ]
        for attempt in range(_JSON_RETRIES + 1):
            msg = generate(
                model=self.llm,
                messages=messages,
                call_name=call_name,
                **self._gen_kwargs(),
            )
            _record_tokens(getattr(self.task, "id", "?"), call_name, msg)
            text = msg.content or ""
            obj = _extract_json(text)
            if obj is not None:
                return obj
            messages = messages + [
                AssistantMessage(role="assistant", content=text),
                UserMessage(
                    role="user",
                    content="That was not valid. Return ONLY a single JSON object, nothing else.",
                ),
            ]
        logger.warning(f"[{call_name}] no valid JSON after {_JSON_RETRIES + 1} attempts.")
        return None

    # -- family / init (MacroStateInitializer = small model) -------------- #

    def _pick_family(self) -> str:
        m = re.match(r"\[([^\]]+)\]", self.task.id or "")
        prefix = m.group(1) if m else ""
        family = _FAMILY_BY_PREFIX.get(prefix)
        if family is None:
            family = next(iter(self.task_spec))  # single-family MVP fallback
        return family

    def _init_task_state(self, family: str) -> dict:
        """MacroStateInitializer: the model fills task_state from the ticket per init_rules."""
        fam_spec = self.task_spec[family]
        schema_fields = (fam_spec.get("task_state_schema", {}) or {}).get("fields", {})
        init_rules = fam_spec.get("init_rules", [])

        fields_block = "\n".join(f"- {k}: {v}" for k, v in schema_fields.items())
        rules_block = "\n".join(f"- {r}" for r in init_rules)
        user = _MACRO_INIT_USER.format(
            fields=fields_block,
            init_rules=rules_block,
            ticket=self.task.ticket or "",
        )
        obj = self._ask_json(_MACRO_INIT_SYSTEM, user, call_name="schema_macro_init")

        ts: dict = {}
        if isinstance(obj, dict):
            # Keep only declared fields; drop anything hallucinated.
            for k in schema_fields:
                if k in obj:
                    ts[k] = obj[k]
        ts.setdefault("resolved", False)  # control layer relies on this being present
        if not ts.get("phone_number"):
            logger.warning("[schema_macro_init] phone_number missing after init — downstream may stall.")
        return ts

    def get_init_state(self, message_history=None) -> SchemaSoloAgentState:
        if message_history is None:
            message_history = []
        family = self._pick_family()
        return SchemaSoloAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            task_state=self._init_task_state(family),
            family=family,
            subtask_done={},
            entered=[],
            pending=None,
            finished=False,
        )

    # -- schema-walk helpers (generic control layer) ---------------------- #

    def _family_spec(self, state):
        return self.task_spec[state.family]

    def _subtask_by_name(self, state, name):
        for st in self._family_spec(state)["base_subtask_sequence"]:
            if st["name"] == name:
                return st
        return None

    def _reground_probe(self) -> Optional[str]:
        """Tool name for the re-grounding probe = CHECK_SERVICE's first base_action.

        Generic across families (all three have a CHECK_SERVICE subtask whose tool is
        the device's primary status read). Returns None if the schema has no such
        subtask_type, in which case re-grounding is skipped.
        """
        spec = self.subtask_spec.get("CHECK_SERVICE", {})
        actions = spec.get("base_actions", [])
        if actions and isinstance(actions[0], dict):
            return actions[0].get("tool")
        return None

    def _on_enter(self, st, state):
        """Apply the subtask's declarative on_enter_reset the first time it activates.

        Replaces the old hardcoded 'reset speed_test after a fix toggle' band-aid:
        entering a fix subtask invalidates the stale diagnostic measurement so its
        done_when_all forces a fresh retest. Fires once per subtask.
        """
        if st["name"] in state.entered:
            return
        state.entered.append(st["name"])
        reset = self.subtask_spec.get(st["subtask_type"], {}).get("on_enter_reset", [])
        for fld in reset:
            state.task_state[_field(fld)] = None

    def _active_subtask(self, state):
        """Select the active subtask.

        Sticky: a subtask started but not yet done stays active even if its entry
        `when` has since flipped (e.g. a fix that satisfied its own entry condition
        must still run its retest). Only when nothing is sticky-active do we pick the
        next subtask by `when` order.
        """
        if state.active_subtask and not state.subtask_done.get(state.active_subtask):
            st = self._subtask_by_name(state, state.active_subtask)
            if st is not None:
                return st
        ts = state.task_state
        for st in self._family_spec(state)["base_subtask_sequence"]:
            if state.subtask_done.get(st["name"]):
                continue
            if eval_pred(st.get("when", "always"), ts):
                state.active_subtask = st["name"]
                self._on_enter(st, state)
                return st
        state.active_subtask = None
        return None

    def _tool_subset(self, subtask, state):
        """Tools the executor may call for this subtask.

        The 'fix already applied -> only retest' restriction is now declarative:
        each fix subtask carries a `retest_guard` {when, restrict_to} in the schema.
        When its `when` holds against the live task_state we drop the flip-tools and
        leave only the retest, so a flip-tool can't fire twice.
        """
        spec = self.subtask_spec[subtask["subtask_type"]]
        names = [ba["tool"] for ba in spec["base_actions"]]
        guard = spec.get("retest_guard")
        if isinstance(guard, dict) and eval_pred(guard.get("when", "never"), state.task_state):
            restrict = guard.get("restrict_to", [])
            names = [n for n in names if n in restrict]
        # Wrap in a schema-sanitizing view so the executor sees model-friendly
        # tool schemas (flattened anyOf/$ref) — see _SanitizedTool.
        return [_SanitizedTool(t) for t in self.tools if t.name in names]

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
        system = _EXECUTOR_SYSTEM.format(ticket=self.task.ticket or "")
        return [
            SystemMessage(role="system", content=system),
            UserMessage(role="user", content=user),
        ]

    # -- StateUpdater (small model) --------------------------------------- #

    def _state_update(self, subtask_type: str, subtask_name: str, tool_name: str, result_text: str, state) -> None:
        """The model parses the tool result into patch_ops; Python validates + applies."""
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
        ops = None
        for attempt in range(_JSON_RETRIES + 1):
            msg = generate(model=self.llm, messages=messages, call_name="schema_state_updater", **self._gen_kwargs())
            _record_tokens(getattr(self.task, "id", "?"), "schema_state_updater", msg)
            text = msg.content or ""
            ops = _parse_patch_ops(text)
            if ops is not None:
                break
            messages = messages + [
                AssistantMessage(role="assistant", content=text),
                UserMessage(role="user", content='Invalid. Return ONLY {"patch_ops": [...]} with set ops on the allowed paths.'),
            ]
        if ops is None:
            logger.warning(f"[schema_state_updater] {subtask_type}: no valid patch_ops; task_state unchanged.")
            _state_log({
                "kind": "update", "task_id": getattr(self.task, "id", "?"),
                "subtask": subtask_name, "subtask_type": subtask_type, "tool": tool_name,
                "result": result_text, "patch_ops": None, "written": [],
                "task_state": dict(state.task_state),
            })
            return
        written = _apply_patch_ops(state.task_state, ops, allowed, fam_enums)
        logger.debug(f"[schema_state_updater] {subtask_type}: wrote {written}")
        _state_log({
            "kind": "update", "task_id": getattr(self.task, "id", "?"),
            "subtask": subtask_name, "subtask_type": subtask_type, "tool": tool_name,
            "result": result_text, "patch_ops": ops, "written": written,
            "task_state": dict(state.task_state),
        })

    def _emit_done(self, state) -> AssistantMessage:
        msg = AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=uuid.uuid4().hex, name=self.STOP_FUNCTION_NAME, arguments={})],
        )
        msg = self._check_if_stop_toolcall(msg)  # -> content=STOP_TOKEN, tool_calls=None
        state.messages.append(msg)
        state.finished = True
        return msg

    # -- main loop --------------------------------------------------------- #

    def generate_next_message(self, message, state):  # type: ignore[override]
        if isinstance(message, UserMessage):
            raise ValueError("SchemaSoloAgent does not support user messages.")

        # 1) Ingest incoming tool result(s) + run the StateUpdater (small model).
        last_result_text = None
        tool_msgs = []
        if isinstance(message, MultiToolMessage):
            tool_msgs = list(message.tool_messages)
        elif isinstance(message, ToolMessage):
            tool_msgs = [message]
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        for tm in tool_msgs:
            state.messages.append(tm)

        if state.pending is not None and tool_msgs:
            pend = state.pending
            match = next((tm for tm in tool_msgs if tm.id == pend.get("tool_call_id")), tool_msgs[-1])
            last_result_text = match.content or ""
            self._state_update(
                pend["subtask_type"], pend["subtask"], pend["tool_name"], last_result_text, state
            )
            # Mark the subtask done if its done_when_all now holds.
            spec = self.subtask_spec.get(pend["subtask_type"], {})
            if eval_pred(spec.get("done_when_all", []), state.task_state):
                state.subtask_done[pend["subtask"]] = True
                if state.active_subtask == pend["subtask"]:
                    state.active_subtask = None
            state.pending = None

        # 2) Decide the next subtask (generic schema walk).
        if state.task_state.get("resolved"):
            _state_log({"kind": "stop", "task_id": getattr(self.task, "id", "?"),
                        "reason": "resolved", "task_state": dict(state.task_state)})
            return self._emit_done(state), state
        subtask = self._active_subtask(state)
        if subtask is None:
            # ② Re-grounding: before giving up with the issue unresolved, re-read
            # ground truth ONCE (the family's primary status probe = CHECK_SERVICE's
            # tool) and refresh task_state. Catches stale-belief dead-ends — e.g. a
            # SIM that flipped to locked_pin after a reseat but whose status was never
            # refreshed, so escalateSim never fired. One-shot (regrounded flag) so it
            # can't loop; skipped if already escalated/resolved. The probe result is
            # parsed by the generic StateUpdater under CHECK_SERVICE's parse_rules.
            reground_probe = self._reground_probe()
            if (
                not state.regrounded
                and reground_probe is not None
                and not state.task_state.get("escalated")
                and not state.task_state.get("resolved")
            ):
                state.regrounded = True
                call_id = uuid.uuid4().hex
                probe_msg = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id=call_id, name=reground_probe, arguments={})],
                )
                state.pending = {
                    "subtask": "__reground__",
                    "subtask_type": "CHECK_SERVICE",
                    "tool_name": reground_probe,
                    "tool_call_id": call_id,
                }
                _state_log({"kind": "reground", "task_id": getattr(self.task, "id", "?"),
                            "probe": reground_probe, "task_state": dict(state.task_state)})
                state.messages.append(probe_msg)
                return probe_msg, state
            _state_log({"kind": "stop", "task_id": getattr(self.task, "id", "?"),
                        "reason": "no_active_subtask", "task_state": dict(state.task_state)})
            return self._emit_done(state), state

        # 3) Executor (small model) picks one tool call from the subtask's subset.
        tool_subset = self._tool_subset(subtask, state)
        messages = self._build_executor_messages(subtask, state, last_result_text)
        assistant_message = generate(
            model=self.llm,
            tools=tool_subset,
            messages=messages,
            tool_choice="required",
            call_name="schema_executor",
            **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_executor", assistant_message)
        if not assistant_message.is_tool_call():
            raise ValueError("SchemaSoloAgent executor must return a tool call.")
        call = assistant_message.tool_calls[0]
        state.pending = {
            "subtask": subtask["name"],
            "subtask_type": subtask["subtask_type"],
            "tool_name": call.name,
            "tool_call_id": call.id,
        }
        _state_log({
            "kind": "exec", "task_id": getattr(self.task, "id", "?"),
            "subtask": subtask["name"], "subtask_type": subtask["subtask_type"],
            "chosen_tool": call.name, "args": call.arguments,
            "task_state": dict(state.task_state),
        })
        state.messages.append(assistant_message)
        return assistant_message, state


# --------------------------------------------------------------------------- #
# Factory  (registration lives in tau2.registry, alongside llm_agent_solo)
# --------------------------------------------------------------------------- #


def create_schema_solo_agent(tools, domain_policy, **kwargs):
    return SchemaSoloAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )


# --------------------------------------------------------------------------- #
# Bare solo BASELINE with tool-schema parity (NO schema control layer)
# --------------------------------------------------------------------------- #
# The reference LLMSoloAgent feeds RAW tool schemas to the model. On local
# qwen3-8b vLLM, Union[Enum,str] params (anyOf/$ref/$defs) make the tool-caller
# emit empty args -> litellm BadRequest -> the whole task dies with
# infrastructure_error (verified: 25/25 bare-solo sims died before the 1st call).
# SchemaSoloAgent dodges this via _SanitizedTool. For a FAIR no-schema baseline
# (the horizon-curve control vs schema_solo_agent) the bare agent needs the SAME
# sanitization — pure tool *callability* parity, NOT a schema/state control layer:
# the env still runs the real tool; only the schema shown to the model is flattened.


class SanitizedLLMSoloAgent(LLMSoloAgent):
    """LLMSoloAgent + tool-schema sanitization, nothing else.

    The bare ReAct-solo baseline that can actually call the telecom tools on a
    qwen tool-parser. No guarded walk, no task_state, no patch_ops — just the
    stock solo agent with model-friendly tool schemas. This is the control arm
    for the long-horizon (success-vs-fault-depth) comparison.
    """

    def __init__(self, tools, domain_policy, task, llm, llm_args=None):
        super().__init__(
            tools=tools, domain_policy=domain_policy, task=task, llm=llm, llm_args=llm_args
        )
        # super().__init__ already ran add_stop_tool() + validate_tools() on the
        # real tools; generate() downstream reads only .name / .openai_schema, so
        # wrapping every tool in the schema-only view here is safe.
        self.tools = [_SanitizedTool(t) for t in self.tools]


def create_llm_solo_agent_sanitized(tools, domain_policy, **kwargs):
    return SanitizedLLMSoloAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
