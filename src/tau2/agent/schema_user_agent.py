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

# L-GROUND: harden the agent↔user instruction SEAM (the M2 failure). The customer
# can only act if the instruction maps to ONE concrete control they can find; the
# default prompt over-translates into invented menu paths ("Settings > About > …")
# and — because the repair subtasks' executor_sys_rules spell out "Step 1: fix /
# Step 2: retest" — the 8B BUNDLES the fix with its retest into one compound message,
# which a one-action-at-a-time customer refuses, looping to max_steps. L-GROUND (a)
# surfaces the action's literal tool+args so it can be named precisely, (b) drops the
# "Step N" scaffolding, and (c) swaps in a system prompt that forbids bundling and
# demands grounded naming. Gated by SCHEMAFLEX_GROUND; OFF = current behavior.
_GROUND_ON = os.environ.get("SCHEMAFLEX_GROUND", "") not in ("", "0", "false", "False")

_INSTRUCT_SYSTEM_GROUND = """You are a friendly telecom support agent talking to a customer who performs device actions themselves (you cannot do it for them). Write ONE short instruction for the SINGLE action given below — and nothing else.

GROUND IT so the customer can actually act — your words must map to one concrete control they can find:
- Use the action's literal arg VALUES verbatim — app names, setting names, permission values. Do NOT rename, capitalise, or prettify them. If app_name=messaging, say "your messaging app" — NOT "Messages" or "your Messages app"; the customer's controls are labelled with those exact strings and a renamed one will NOT be found.
- Name the exact setting/toggle. Examples: toggle_wifi_calling → "turn off Wi‑Fi Calling"; grant_app_permission (app_name=messaging, permission=sms) → "in your messaging app, grant the SMS permission"; reset_apn_settings → "reset your APN settings to default"; toggle_roaming → "turn Data Roaming on".
- Do NOT send the customer hunting through invented menu paths for a value the action names directly.
- If the action only CHECKS or reads something (a check_… , run_… , get_… , or can_… tool), ask them to LOOK at it and report what it shows / whether it works — do NOT tell them to enable, change, or toggle anything (there may be no such setting).

ONE ACTION ONLY — this is the single most common failure:
- Do the ONE action above. If the guidance mentions a later check, retest, reboot, or "Step 2" (e.g. "then check if you can send MMS"), DO NOT include it — that is a separate later turn.
- Never bundle two settings/actions, or an action plus a verification, in one message. Many customers refuse a multi-part request and the whole turn is wasted.
- End by asking them to report what they see or what happened.

One or two sentences. No markdown, no lists, no preamble."""

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
# DIAG (retest-yield): break REPLAN's wrong-repair thrash on mms. When the speculative
# permission repairs (fixStoragePerm/fixSmsPerm, gated `mms_*_perm != true`, eligible via
# the None default) fire but DON'T fix it, the goal retest (tier-3) keeps being re-picked
# over the tier-2 diagnostics — so the agent loops retestMms and never reaches
# checkWifiCalling/checkNetworkMode to find the REAL cause. DIAG demotes a goal retest that
# has already run-and-failed while an informative diagnostic still pends, so the diagnostic
# runs and reveals the correct repair. It does NOT touch repair priority — tasks the
# speculative repair actually fixes still close fast. Gated; OFF = REPLAN unchanged.
_DIAG_ON = os.environ.get("SCHEMAFLEX_DIAG", "") not in ("", "0", "false", "False")
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

# L-SHOTGUN: mms speculative fixed-order repair. The mms root cause is undiagnosable
# from the complaint — 6 distinct faults (network_mode / wifi_calling / apn / sms_perm /
# storage_perm / both) share a BYTE-IDENTICAL "can't send MMS for the past few hours"
# ticket and identical known_info, so any pre-diagnosis root-cause prediction is a 1-of-6
# guess (this is why L-SELECT, a strong-model supervisor, couldn't pick correctly — the
# signal simply isn't in context yet). And the 40-step budget can't fit
# diagnose→fix→retest per candidate. BUT: (a) reward is pure ENV_ASSERTION on
# can_send_mms — no DB/ACTION basis (verified across all 8 mms tasks), so extra/odd
# device mutations are NOT penalized; (b) _can_send_mms is an AND of every condition, so
# making them ALL true necessarily succeeds regardless of which one was broken; (c) every
# fix is idempotent or verified-harmless — set_network_mode/reset_apn/grant_perm/reboot
# are idempotent, and a blind toggle_wifi_calling is safe on ALL 6 mms tasks because
# wifi_calling_mms_over_wifi defaults False everywhere except bad_wifi_calling (the AND
# makes the enabled-flip inert). So once basic connectivity holds, SKIP all diagnostics
# and fire every repair in a fixed safe order, then one can_send_mms retest → L-CLOSE.
# Orchestration-only, zero extra LLM cost; sidesteps the root-cause discrimination that
# L-SELECT proved unsolvable. Gated by SCHEMAFLEX_SHOTGUN; OFF = byte-identical.
_SHOTGUN_ON = os.environ.get("SCHEMAFLEX_SHOTGUN", "") not in ("", "0", "false", "False")
# Fixed plan = family-sequence subtask names (indexed, so retestMms may repeat). We fire
# each subtask's base_action[0] (the mutating fix), bypassing its diagnostic `when`.
# BATCHED RETEST: the 4 immediate-effect fixes (perms/network/wifi all take effect at
# once) run first, then a retest — which CLOSES 5 of 6 mms faults (storage/sms/both-perm,
# network, wifi) the moment can_send_mms flips true, so we never pay for the apn tail.
# Only an apn fault leaves the first retest false; we then reset_apn (flags reset_at_reboot)
# → reboot (executes the reset → restores mmsc) → second retest. Order within the batch is
# harmless (all idempotent / verified-safe). This early close is what keeps the plan inside
# the 40-step budget: the v1 single-trailing-retest plan fixed the fault mid-sequence but
# ran out of steps before ever measuring can_send_mms, so L-CLOSE never fired (0/6).
# `checkMmsPerms` leads: a blind grant to a COLD (diagnosis-skipped) customer gets balked
# ("how do I do that?"), and a balk skips the fix → perm tasks regress (the directive-first
# attempt to fix this by phrasing failed: verbose Settings paths just invite narration).
# The fix is what the NORMAL walk does — run check_app_permissions FIRST: it (a) PRIMES the
# customer (navigates them to the permissions screen, so the subsequent grant lands first
# try) and (b) reveals mms_*_perm so a perm already granted is SKIPPED (`_SHOTGUN_SKIP_WHEN`)
# — which also frees the budget that lets the apn tail fit (a non-perm fault skips both
# grants). It's a priming/routing check, NOT the root-cause discrimination L-SELECT/DIAG
# failed at (shotgun still blind-fixes network/wifi/apn).
_SHOTGUN_PLAN = [
    "checkMmsPerms",      # check_app_permissions(messaging) — PRIME user + reveal mms_*_perm
    "fixStoragePerm",     # grant storage — SKIP if mms_storage_perm already true
    "fixSmsPerm",         # grant sms     — SKIP if mms_sms_perm already true
    "fixNetworkModeMms",  # set_network_mode_preference(4g_5g)  — idempotent
    "fixWifiCalling",     # toggle_wifi_calling()              — verified safe ∀ mms
    "retestMms",          # can_send_mms() → closes perms/network/wifi faults
    "resetApnMms",        # reset_apn_settings() — apn fault only past here
    "rebootMms",          # reboot_device() — executes APN reset → restores mmsc
    "retestMms",          # can_send_mms() → closes the apn fault
]
# A plan step whose fix is already satisfied (perm granted) is skipped without firing —
# saves the turn AND lets the apn path fit the 40-step budget on non-perm faults.
_SHOTGUN_SKIP_WHEN = {
    "fixStoragePerm": "mms_storage_perm",
    "fixSmsPerm": "mms_sms_perm",
}
# A SHOTGUN step advances only once its fix actually LANDS (the user executed the tool, so
# its parse wrote task_state). A cold user who skipped diagnosis often balks on the first
# bare instruction ("grant the storage permission" → "how do I do that?"); without sticking
# we'd skip the fix forever (the v2 perm-task regression). So re-instruct the SAME step (a
# balk costs only 2 messages) up to this many times, with a directive focus hint on retry,
# before giving up and advancing — mirroring the normal walk's stickiness.
_SHOTGUN_RETRY_CAP = 3

# L-ELICIT: a repair can be gated on a CONTEXT PRECONDITION the schema assumes was
# pre-filled from the ticket but which is actually obtainable by ASKING the user — e.g.
# makePayment is gated `pay_allowed == true`, and pay_allowed is set ONLY by init_rules
# from the ticket text. When the authorization is conditional/interactive ("if asked, you
# accept") rather than stated upfront, the field stays null, the repair is dead-gated, and
# the agent gives up → transfer → reward 0 (overdue_bill_suspension). The GENERAL criterion
# (no hard-coded field): a field that (a) appears in some repair's `when`, (b) is produced
# by NO subtask (so it's a ticket-context field, not a diagnostic signal), and (c) is still
# null, can only be obtained by asking. So at the give-up point, if a pending on-chain
# repair is blocked solely by such a field, ASK the user a yes/no question, parse the
# answer, set the field, and let the flow cascade. Covers pay_allowed / is_abroad / … with
# one mechanism. Gated by SCHEMAFLEX_ELICIT; OFF = byte-identical.
_ELICIT_ON = os.environ.get("SCHEMAFLEX_ELICIT", "") not in ("", "0", "false", "False")

# --------------------------------------------------------------------------- #
# L-SEQ: ordered multi-action subtask sequencing guard.
#
# Motivating bug (overdue_bill_suspension, MAKE_PAYMENT = [check_payment_request,
# make_payment]): the read action's `when` (... bill_paid != true) STAYS true after it
# runs, so `_next_action` re-selects the read forever and the terminal write make_payment
# (whose `when` is the prose "after check_payment_request succeeded", never eval-true) is
# never reached. Worse, the StateUpdater can mis-parse the read's output into the done-field
# (bill_paid=False from "You HAVE a payment request"), satisfying done_when_all (bill_paid !=
# null) and closing the subtask before make_payment ever fires — chain dies, agent escalates.
#
# SEQ enforces the subtask's own stated contract ("call each at most once, in order"):
#   R-skip      `_next_action` skips a NON-terminal base_action already dispatched this
#               activation (the terminal is never skipped, so a balked retest stays
#               re-instructable), so the read can't loop and the terminal gets selected.
#   R-doneguard `_mark_done_if_complete` withholds completion of a multi-action subtask until
#               its TERMINAL base_action has dispatched, so a hallucinated parse of a
#               non-terminal read can't close it early. (Inert for FIX_*: their done-field is
#               produced only by the terminal retest, so done_when_all ⟹ terminal dispatched.)
# Gated by SCHEMAFLEX_SEQ; OFF = byte-identical.
_SEQ_ON = os.environ.get("SCHEMAFLEX_SEQ", "") not in ("", "0", "false", "False")

# --------------------------------------------------------------------------- #
# Replan-failure fixes (diagnosed on strat90_replan = 60/90 user-mode). The
# schema was authored from the SOLO tech-support doc, where the agent runs every
# device tool itself and reads the result in the same turn. In USER-mode those
# tools are user-side: the agent must instruct → read the customer's reply →
# verify. The solo-authored schema never modeled that seam. Each flag below is an
# independent lever; OFF = byte-identical (project ablation convention).
# --------------------------------------------------------------------------- #
# L-PROV: action-level provenance for the family GOAL field. FIX subtasks are
# "fix + retest" compounds (e.g. FIX_NETWORK_MODE = [set_network_mode, run_speed_test])
# and allow-list the goal field (speed_test / can_send_mms). The StateUpdater
# reacting to the reply to the FIX action (set_network_mode) writes
# speed_test='excellent' BEFORE run_speed_test is ever issued → phantom-satisfies
# success_when_all → the proactive close latches `resolved` → the agent freezes on
# the closing line → the customer (who never ran a speed test) refuses → OOS /
# hard-zero (4/30 mobile_data fails). PROV drops a write to a goal field unless
# the just-instructed action IS that field's dedicated verify tool, forcing the
# real run_speed_test / can_send_mms round-trip the user-mode doc requires.
_PROV_ON = os.environ.get("SCHEMAFLEX_PROV", "") not in ("", "0", "false", "False")

# L-UNLATCH: `resolved` is a latch with no un-set path — once the proactive close
# sets it, every later turn returns the canned closing line to max_steps. If the
# customer keeps talking after a PROACTIVE close (they didn't ###STOP### → they do
# not agree it is fixed), clear `resolved` + the goal fields and re-open
# verification so the walk resumes. Defense-in-depth behind PROV.
_UNLATCH_ON = os.environ.get("SCHEMAFLEX_UNLATCH", "") not in ("", "0", "false", "False")

# L-WATCHDOG: episode-level no-progress guard. The per-subtask `stuck` counter is
# keyed on subtask NAME, so REPLAN-thrash (a different subtask each turn) never
# accumulates it → give-up never fires → max_steps. WATCHDOG escalates after
# _WATCHDOG_REPEAT identical consecutive instructions (verbatim deadlock) and
# after _WATCHDOG_STALL turns with no goal-field advance (thrash / depth-collapse).
# It bounds wasted turns (an efficiency / $ guard); it does not by itself recover
# reward on a task that was not converging.
_WATCHDOG_ON = os.environ.get("SCHEMAFLEX_WATCHDOG", "") not in ("", "0", "false", "False")
_WATCHDOG_REPEAT = int(os.environ.get("SCHEMAFLEX_WATCHDOG_REPEAT", "3"))  # nth identical re-emit → stop
_WATCHDOG_STALL = int(os.environ.get("SCHEMAFLEX_WATCHDOG_STALL", "15"))   # frozen turns (no goal/subtask motion) → stop

# L-REPLY: feed the customer's last reply into _phrase_instruction so a re-ask
# answers their concrete question (e.g. supply the literal app name "messaging")
# instead of re-emitting verbatim; also pins the literal app-name grounding on the
# NORMAL walk (previously shotgun-only) for the mms permission deadlock.
_REPLY_ON = os.environ.get("SCHEMAFLEX_REPLY", "") not in ("", "0", "false", "False")

# L-HILO: high-low collaboration. The 8B StateUpdater is the proven bottleneck —
# an offline study (gpt-5.4 oracle) found ~19% of high-consequence extractions are
# CONSEQUENTIALLY wrong (writes resolved from a failure, mis-targets a field, drops
# a co-field, mis-grades a value), and these are CONFIDENT errors (self-consistency
# sampling caught only ~13%, so SAGE-style model-uncertainty does NOT fit). So HILO
# routes only the LOAD-BEARING extractions to a big model: (Tier 1) the terminal
# VERIFY/RETEST extractions that gate the close, always; (Tier 2) elsewhere, only
# when a cheap structural check flags an obvious 8B error. The 8B still does the
# instruction phrasing (Executor) and the majority of extractions — it stays the
# workhorse; the big model is a sparse corrector. OFF = byte-identical.
_HILO_ON = os.environ.get("SCHEMAFLEX_HILO", "") not in ("", "0", "false", "False")
_HILO_LLM = os.environ.get("SCHEMAFLEX_HILO_LLM", "openai/gpt-5.4")
_HILO_BASE = os.environ.get("SCHEMAFLEX_HILO_BASE", "")  # "" → litellm uses OPENAI_API_BASE (.env proxy)
_HILO_MAX = int(os.environ.get("SCHEMAFLEX_HILO_MAX", "12"))  # per-episode big-model correction budget
_HILO_VERIFY = set(
    (os.environ.get("SCHEMAFLEX_HILO_VERIFY")
     or "VERIFY_SPEED,VERIFY_MMS,VERIFY_SERVICE,RETEST_MMS,"
        "FIX_STORAGE_PERM,FIX_SMS_PERM,FIX_NETWORK_MODE_MMS,FIX_WIFI_CALLING").split(",")
)  # Tier-1 load-bearing subtasks: always big-model extraction. Covers the terminal
#    verify/retest (close-critical) AND the mms repair zone (perm/wifi-calling/net-mode
#    extractions that feed can_send_mms), where the offline study found the 8B's
#    catastrophic drops (missed mms_sms_perm, wrong-field). Tunable via env.

# Cheap structural-error signals for the Tier-2 gate (refined on the offline study:
# ~47% hit / 3% FP on the obvious bias errors the 8B makes). They catch confident
# mistakes self-consistency can't; the semantic misses (co-field/value-grade) are
# covered by Tier-1 consequence routing instead.
_HILO_RESULT_KW = re.compile(
    r"\b(mbps|signal|shows?|status|bar|excellent|good|fair|poor|no connection|connection|"
    r"fail(ed)?|active|connected|sim|airplane|wi-?fi|roaming|granted|permission|sent|deliver|"
    r"message|did that|now|i (turned|ran|checked|granted|reset|reboot|enabled|disabled|tried|set|gave|changed))\b",
    re.I,
)
_HILO_PURE_Q = re.compile(
    r"\b(could you|can you|which|what do you mean|not sure which|before i|please (tell|specify|clarify)|"
    r"do you want|should i|tell me which|which (exact )?app)\b",
    re.I,
)
_HILO_FAIL = re.compile(
    r"\b(still (can'?t|not|isn'?t)|can'?t send|did(n'?t| not) (work|send)|failed|no connection|not excellent)\b",
    re.I,
)
_HILO_SUCCESS_VALS = {"speed_test": {"excellent"}, "can_send_mms": {True, "true", "True"},
                      "service_status": {"connected"}, "resolved": {True, "true", "True"}}

_ELICIT_SYSTEM = """You are a friendly telecom support agent. Before you can apply a fix, you need ONE piece of authorization or context from the customer that only they can give (e.g. permission to charge an overdue bill, or whether they are currently travelling abroad). Write ONE short, friendly yes/no question that obtains exactly that — name the concrete thing (the specific bill, the specific action). Do NOT explain steps or list options; just ask the single question. Output only the question text."""

_ELICIT_USER = """You need the customer's: {field}
Granting it unblocks this fix: {repair_goal}
Overall goal of the call: {task_goal}
Current machine state (use it for concrete details like the bill id): {task_state}
Write the single short yes/no question now."""

_AUTH_SYSTEM = """The customer was just asked a yes/no authorization/context question. Read their reply and answer with EXACTLY one lowercase word:
- yes  — they agree / accept / authorize / confirm it is true
- no   — they decline / refuse / say it is false
- unclear — anything else (a question back, hesitation, off-topic)
Output only that one word."""

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
    repair_attempted: bool = False  # a mutating device/backend tool was dispatched this episode (L-CLOSE guard)
    # what we are waiting on: 'tool' (env result), 'user' (reply to an instruction),
    # 'await_phone' (reply to the phone-number question), or None.
    pending_kind: Optional[str] = None
    shotgun_idx: int = 0  # L-SHOTGUN: index of the next mms repair-plan step to fire
    shotgun_retry: int = 0  # L-SHOTGUN: re-instruct count for the current step (until it lands)
    elicited: list = []  # L-ELICIT: context-precondition fields already asked of the user
    acted: dict = {}  # L-SEQ: subtask name -> base_action tools already dispatched this activation
    closed_proactively: bool = False  # L-UNLATCH: `resolved` was set by the proactive close, not a VERIFY
    last_instr: Optional[str] = None  # L-WATCHDOG: last instruction text emitted (verbatim-repeat detector)
    instr_repeat: int = 0  # L-WATCHDOG: consecutive identical re-emits
    stall_turns: int = 0  # L-WATCHDOG: user-reply turns since the goal field last advanced
    last_user_reply: Optional[str] = None  # L-REPLY: the customer's most recent free-text reply
    hilo_calls: int = 0  # L-HILO: big-model StateUpdater corrections spent this episode (budget cap)


class SchemaUserAgent(SchemaSoloAgent):
    """Non-solo schema agent (see module docstring)."""

    # Capture the per-sim seed (orchestrator calls set_seed once per simulation; base impl
    # is an optional no-op). seed is distinct per trial → used to key the token log per-SIM
    # (task_id, seed) so num-trials>1 trials stay separable for pass^4 decay analysis.
    def set_seed(self, seed: int) -> None:
        self._seed = seed

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
        if _SEQ_ON:
            # L-SEQ R-skip: skip a non-terminal base_action already dispatched this activation,
            # so a READ-ONLY gate whose `when` stays true (e.g. check_payment_request: bill_paid
            # != true — checking never sets bill_paid) can't loop and starve the terminal write.
            # Restricted to read-only tools: a read lands deterministically on dispatch (the user
            # just reports what they see), so at-most-once is safe. MUTATING device actions can
            # balk (the user doesn't perform them) and MUST stay re-instructable — never skip
            # them, or a balked flip is lost and the subtask loops on its retest forever
            # (observed killing mms repairs in the unscoped version). The terminal action is also
            # never skipped (a balked retest must remain re-instructable).
            acted = set(state.acted.get(subtask["name"], []))
            last_tool = bas[-1].get("tool")
            for ba in bas:
                t = ba.get("tool") or ""
                if t in acted and t != last_tool and _READONLY_TOOL_RE.match(t):
                    continue
                if eval_pred(ba.get("when", "always"), state.task_state):
                    return ba
            return bas[-1]
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

    def _verify_field_tools(self) -> dict:
        """L-PROV map: {goal_field -> the single tool that actually measures it},
        auto-derived from the schema (each VERIFY_* subtask with exactly one
        base_action whose tool is a dedicated measurement read). Restricted to
        run_speed_test / can_send_mms: those are pure measurements no FIX surfaces
        as a side effect. service_status is intentionally excluded — fixes
        legitimately surface it via the status bar, and the service family is at
        ceiling, so gating it would only add risk. Derived, not hand-mapped, so it
        carries to a re-generated schema (auto schema-gen north star)."""
        cache = getattr(self, "_vft_cache", None)
        if cache is not None:
            return cache
        m = {}
        for name, spec in self.subtask_spec.items():
            if not str(name).startswith("VERIFY"):
                continue
            bas = spec.get("base_actions") or []
            if len(bas) != 1:
                continue
            tool = bas[0].get("tool")
            if tool not in ("run_speed_test", "can_send_mms"):
                continue
            for f in self._dwa_fields(name):
                m[f] = tool
        self._vft_cache = m
        return m

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

    # -- L-SHOTGUN: mms speculative fixed-order repair ------------------- #
    def _shotgun_ready(self, state):
        """Active as soon as the line is resolved — we deliberately SKIP the
        checkService/diagnoseData prerequisites (≈8 messages of network-check + speed-test)
        because none of the mms fixes or the can_send_mms close depend on service_status or
        speed_test, and that overhead is what blew the 40-step budget in v1. The step-3
        resolved-check already gates closure, so no `resolved` test is needed here."""
        if not _SHOTGUN_ON or state.family != "mms":
            return False
        return state.task_state.get("line_id") is not None

    def _shotgun_next(self, state):
        """First un-skipped plan step at/after the current index as (subtask,
        base_action[0]); (None, None) when the plan is exhausted (→ fall back to the normal
        walk / L-CLOSE). Progress is tracked by state.shotgun_idx, NOT done_when_all — those
        aspects depend on diagnostic fields SHOTGUN skips, so they never close on their own;
        index also lets retestMms repeat (batched-retest plan). A step whose fix is already
        satisfied (`_SHOTGUN_SKIP_WHEN`, e.g. perm already granted per the checkMmsPerms
        read) is advanced past WITHOUT firing — saves the turn and frees apn-tail budget."""
        ts = state.task_state
        idx = state.shotgun_idx or 0
        while idx < len(_SHOTGUN_PLAN):
            name = _SHOTGUN_PLAN[idx]
            skip_field = _SHOTGUN_SKIP_WHEN.get(name)
            if skip_field is not None and ts.get(skip_field) is True:
                idx += 1
                continue  # fix already satisfied — skip without consuming a turn
            st = self._subtask_by_name(state, name)
            bas = self.subtask_spec.get(st["subtask_type"], {}).get("base_actions", []) if st else []
            if not bas:
                idx += 1
                continue
            state.shotgun_idx = idx  # persist any skips so INGEST's +1 lands correctly
            return st, bas[0]
        state.shotgun_idx = idx
        return None, None

    # -- L-ELICIT: ask the user for a null context-precondition --------- #
    def _context_precondition_fields(self, state):
        """Fields a subtask `when` depends on that NO subtask produces — the schema assumes
        the ticket pre-filled them (pay_allowed / is_abroad / …). Cached per family. These
        are the only fields that, when null, can be obtained by ASKING the user (a diagnostic
        field would instead be obtained by running its check_* tool)."""
        cache = getattr(self, "_ctx_field_cache", None)
        if cache is None:
            cache = self._ctx_field_cache = {}
        fam = state.family
        if fam not in cache:
            seq = self._family_spec(state)["base_subtask_sequence"]
            produced, used = set(), set()
            for st in seq:
                produced |= self._produces(st["subtask_type"])
                used |= set(self._pred_fields(st.get("when", "always")))
            structural = {"goal", "customer_name", "phone_number", "resolved",
                          "candidate_line_ids", "customer_id", "line_id", "escalated"}
            cache[fam] = (used - produced) - structural
        return cache[fam]

    def _elicit_target(self, state):
        """(field, target_bool, repair_subtask) to ask the user for, or None. Picks a
        pending, on-chain REPAIR that is NOT eligible only because a single context-
        precondition field (no producer) is still null — asking the user for it unblocks
        the repair and the downstream flow cascades."""
        if not _ELICIT_ON:
            return None
        ctx = self._context_precondition_fields(state)
        if not ctx:
            return None
        ts = state.task_state
        asked = set(state.elicited or [])
        idx = self._causal_index(state)
        for st in self._family_spec(state)["base_subtask_sequence"]:
            t = st["subtask_type"]
            when = st.get("when", "always")
            if state.subtask_done.get(st["name"]) or not idx["is_repair"][t]:
                continue
            if eval_pred(when, ts):
                continue  # already eligible
            # Candidate context-precondition atoms: `field == true/false`, field is a
            # no-producer context field, currently null, not yet asked.
            for m in re.finditer(r"task_state\.([a-zA-Z_]\w*)\s*==\s*(true|false)", when):
                f, v = m.group(1), m.group(2)
                if f not in ctx or ts.get(f) is not None or f in asked:
                    continue
                tv = (v == "true")
                # Tight gate (fixes over-firing): elicit f ONLY if setting it to its target
                # makes THIS repair eligible right now — i.e. every OTHER atom already
                # holds. Otherwise the repair isn't actually one step away (e.g. makePayment
                # in a non-billing task where payment_requested is still null), and asking
                # "do you authorize payment?" is a nonsensical derail.
                probe = dict(ts)
                probe[f] = tv
                if eval_pred(when, probe):
                    return f, tv, st
        return None

    def _phrase_elicitation(self, field, repair, state) -> str:
        import json
        spec = self.subtask_spec.get(repair["subtask_type"], {})
        user = _ELICIT_USER.format(
            field=field,
            repair_goal=spec.get("goal_template", ""),
            task_goal=state.task_state.get("goal", ""),
            task_state=json.dumps(state.task_state, ensure_ascii=False),
        )
        msg = generate(
            model=self.llm,
            messages=[SystemMessage(role="system", content=_ELICIT_SYSTEM),
                      UserMessage(role="user", content=user)],
            call_name="schema_executor", **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_executor", msg, sim_seed=getattr(self, "_seed", None))
        return (msg.content or "").strip() or "Before I proceed, do you authorize this action?"

    def _parse_authorization(self, text: str) -> str:
        """Classify a yes/no authorization reply → 'yes' | 'no' | 'unclear'."""
        msg = generate(
            model=self.llm,
            messages=[SystemMessage(role="system", content=_AUTH_SYSTEM),
                      UserMessage(role="user", content=text)],
            call_name="schema_state_updater", **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_state_updater", msg, sim_seed=getattr(self, "_seed", None))
        w = (msg.content or "").strip().lower()
        return "yes" if w.startswith("yes") else ("no" if w.startswith("no") else "unclear")

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

    def _retest_exhausted(self, state, idx, needed):
        """DIAG retest-yield: the family goal-core field has already been MEASURED as
        still failing (a repair ran, the retest confirmed it didn't work) AND an un-run,
        live, informative diagnostic still pends. Re-running the retest is then futile —
        yield to the diagnostic that may reveal a DIFFERENT root cause/repair, instead of
        looping the goal retest to max_steps (the mms wrong-repair thrash: fixStoragePerm
        →fixSmsPerm→retestMms loop that never reaches checkWifiCalling). When the goal is
        actually met, or no diagnostic is left, returns False so the normal tier-3 closer
        (and the give-up gate) apply unchanged. Does NOT touch repair priority, so the
        fast path on tasks the speculative repair actually fixes is preserved."""
        ts = state.task_state
        goal_core = idx["goal_fields"] - {"resolved"}
        if not any(ts.get(f) is not None for f in goal_core):
            return False  # goal never measured yet → let the retest run
        swa = self._family_spec(state).get("success_when_all", [])
        if not swa or eval_pred(swa, ts):
            return False  # goal actually MET → let it close, don't yield
        seq = self._family_spec(state)["base_subtask_sequence"]
        for st in seq:
            if idx["is_repair"][st["subtask_type"]] or state.subtask_done.get(st["name"]):
                continue
            if self._informative(state, idx, needed, st) and self._when_live(st.get("when", "always"), ts):
                return True  # a fresh diagnostic can still reveal a different repair
        return False

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
                if _DIAG_ON and self._retest_exhausted(state, idx, needed):
                    tier = 1  # already retested & still failing → yield to pending diagnostics
                else:
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
            _record_tokens(tid, "schema_supervisor", msg, sim_seed=getattr(self, "_seed", None))
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
        _record_tokens(getattr(self.task, "id", "?"), "schema_macro_init", msg, sim_seed=getattr(self, "_seed", None))
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

        def _act_line(ba):
            # L-GROUND surfaces the literal tool args (app_name=…, permission=…) so the
            # instruction can name the exact control; OFF keeps the original tool: when.
            argstr = ""
            if _GROUND_ON and isinstance(ba.get("args"), dict) and ba["args"]:
                argstr = " (" + ", ".join(f"{k}={v}" for k, v in ba["args"].items()) + ")"
            return f"- {ba.get('tool')}{argstr}: {ba.get('when', '')}"

        if chosen is not None:
            actions = _act_line(chosen)
        else:
            actions = "\n".join(_act_line(ba) for ba in spec.get("base_actions", []))
        rule_list = spec.get("executor_sys_rules", [])
        if _GROUND_ON:
            # drop the "Step N:" scaffolding — it makes the 8B bundle the fix with its
            # retest/reboot into one compound instruction (the M2 loop). The single
            # action is already pinned by `actions`.
            rule_list = [r for r in rule_list if not str(r).strip().lower().startswith("step ")]
        rules = "\n".join(f"- {r}" for r in rule_list)
        if focus:  # SAGE: surface the closed option set on a re-ask
            rules = f"{rules}\n{focus}" if rules else focus
        # L-GROUND also drops the goal_template here: it states the WHOLE multi-step
        # fix ("fix via toggle_data_saver_mode, run_speed_test"), which leaks the
        # later retest into a single-action instruction → the 8B bundles "toggle +
        # run speed test", the one-action customer refuses, and the agent loops to
        # max_steps with the retest never run (the type-(a) "repair done, no close").
        goal = "" if _GROUND_ON else spec.get("goal_template", "")
        user = _INSTRUCT_USER.format(
            subtask_name=subtask["name"],
            subtask_type=subtask["subtask_type"],
            goal=goal,
            actions=actions,
            rules=rules,
        )
        msg = generate(
            model=self.llm,
            messages=[
                SystemMessage(role="system", content=_INSTRUCT_SYSTEM_GROUND if _GROUND_ON else _INSTRUCT_SYSTEM),
                UserMessage(role="user", content=user),
            ],
            call_name="schema_executor",
            **self._gen_kwargs(),
        )
        _record_tokens(getattr(self.task, "id", "?"), "schema_executor", msg, sim_seed=getattr(self, "_seed", None))
        return (msg.content or "").strip() or "Could you try that step and tell me what you see?"

    def _hi_gen_kwargs(self) -> dict:
        """Generation kwargs for the big-model corrector: temp 0, proxy base. Unlike
        _gen_kwargs it does NOT carry the local-vLLM api_base or the Qwen
        thinking-disable extra_body — the big model is reached via OPENAI_API_BASE
        (.env proxy) unless SCHEMAFLEX_HILO_BASE overrides it."""
        kw = {"temperature": 0.0}
        if _HILO_BASE:
            kw["api_base"] = _HILO_BASE
        return kw

    def _run_state_updater(self, messages, model, gen_kwargs, call_name, tid):
        """One StateUpdater extraction with JSON-retry. Returns parsed patch_ops or None."""
        ops, msgs = None, messages
        for _ in range(_JSON_RETRIES + 1):
            msg = generate(model=model, messages=msgs, call_name=call_name, **gen_kwargs)
            _record_tokens(tid, call_name, msg, sim_seed=getattr(self, "_seed", None))
            ops = _parse_patch_ops(msg.content or "")
            if ops is not None:
                break
            msgs = msgs + [
                AssistantMessage(role="assistant", content=msg.content or ""),
                UserMessage(role="user", content='Invalid. Return ONLY {"patch_ops": [...]} on the allowed paths.'),
            ]
        return ops

    def _extraction_suspect(self, reply, ops):
        """Cheap structural check (no LLM): does the 8B's patch_ops look like a
        CONFIDENT mis-extraction? Returns a tag or None. Refined on the offline study
        to ~47% hit / 3% FP on the obvious bias errors."""
        norm = {}
        for o in ops or []:
            if isinstance(o, dict) and o.get("op") == "set":
                norm[_field(str(o.get("path", "")))] = o.get("value")
        rl = reply or ""
        if norm.get("resolved") in (True, "true", "True"):
            return "resolved_true"              # StateUpdater must never set resolved
        has_q = bool(_HILO_PURE_Q.search(rl))
        has_obs = bool(_HILO_RESULT_KW.search(rl))
        if norm and has_q and not has_obs:
            return "nonanswer_write"            # wrote from a pure question/refusal
        if not norm and has_obs and not has_q:
            return "empty_on_substantive"       # under-extraction from a real observation
        if _HILO_FAIL.search(rl) and any(norm.get(k) in v for k, v in _HILO_SUCCESS_VALS.items()):
            return "false_success"              # success value while the reply states failure
        return None

    def _hilo_route(self, subtask_type, reply, ops_8b, state):
        """Decide whether to spend a big-model correction on this extraction.
        Returns a route tag or None. Budget-capped per episode (8B stays workhorse)."""
        if state.hilo_calls >= _HILO_MAX:
            return None
        if subtask_type in _HILO_VERIFY:
            return "verify"                     # Tier 1: close-critical extraction, always
        return self._extraction_suspect(reply, ops_8b)  # Tier 2: structural gate elsewhere

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
        if _PROV_ON:
            # Action-level provenance: a goal/verification field may only be written
            # by the StateUpdater reacting to its OWN measurement tool. Reacting to a
            # FIX action (set_network_mode, grant_app_permission, …) it is dropped, so
            # speed_test/can_send_mms cannot be phantom-set before the retest runs.
            vft = self._verify_field_tools()
            block = {f for f, t in vft.items() if action_name != t}
            if block:
                allowed = [a for a in allowed
                           if not (isinstance(a, dict) and _field(str(a.get("path", ""))) in block)]
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
        tid = getattr(self.task, "id", "?")
        # 8B does the extraction (workhorse). L-HILO: on load-bearing extractions
        # (Tier-1 terminal verify, or Tier-2 structural-flagged 8B errors) a big model
        # re-extracts and overrides — sparse, budget-capped, token-split for $.
        ops = self._run_state_updater(messages, self.llm, self._gen_kwargs(), "schema_state_updater", tid)
        hilo_route = None
        if _HILO_ON:
            hilo_route = self._hilo_route(subtask_type, reply_text, ops or [], state)
            if hilo_route:
                ops_hi = self._run_state_updater(
                    messages, _HILO_LLM, self._hi_gen_kwargs(), "schema_state_updater_hi", tid)
                if ops_hi is not None:
                    state.hilo_calls += 1
                    _state_log({"kind": "hilo", "task_id": tid, "subtask_type": subtask_type,
                                "action": action_name, "route": hilo_route,
                                "ops_8b": ops, "ops_hi": ops_hi, "n_hilo": state.hilo_calls})
                    ops = ops_hi
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
            if _SEQ_ON:
                # L-SEQ R-doneguard: a multi-action subtask whose TERMINAL action is a MUTATING
                # write isn't done until that write has dispatched, so a hallucinated parse of a
                # non-terminal read (check_payment_request → bill_paid=False) can't close it
                # before make_payment runs. Gated on a mutating terminal: FIX_* terminals are
                # the read-only retest (can_send_mms / run_speed_test), whose field done_when_all
                # already requires — so the guard is off there (original behavior, no loop risk).
                bas = spec.get("base_actions", []) or []
                last_tool = (bas[-1].get("tool") or "") if bas else ""
                if (len(bas) > 1 and _MUT_TOOL_RE.match(last_tool)
                        and last_tool not in state.acted.get(pend["subtask"], [])):
                    return
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
            state.last_user_reply = text  # L-REPLY: surface to the next instruction phrasing
            if state.pending_kind == "await_phone":
                self._parse_phone(text, state)
                state.pending = None
                state.pending_kind = None
            elif state.pending_kind == "await_elicit":
                # L-ELICIT: parse the yes/no authorization reply into the context field.
                pend = state.pending or {}
                f, v = pend.get("elicit_field"), pend.get("elicit_value")
                ans = self._parse_authorization(text)
                if ans == "yes":
                    state.task_state[f] = v
                elif ans == "no":
                    state.task_state[f] = (not v) if isinstance(v, bool) else False
                # 'unclear' → leave null (the `elicited` guard still prevents a re-ask)
                _state_log({"kind": "elicit_reply", "task_id": tid, "field": f,
                            "answer": ans, "value": state.task_state.get(f),
                            "task_state": dict(state.task_state)})
                state.pending = None
                state.pending_kind = None
            elif state.pending is not None and state.pending_kind == "user":
                pend = state.pending
                _gf = self._pred_fields(self._family_spec(state).get("success_when_all", []))
                _goal_before = {f: state.task_state.get(f) for f in _gf}
                _sk_done_before = state.subtask_done.get(pend["subtask"], False)
                written = self._update_from_user_reply(
                    pend["subtask_type"], pend["subtask"], pend["tool_name"], text, state
                )
                self._mark_done_if_complete(pend, state)
                # L-SHOTGUN: advance the plan only when the fix LANDED (written) or we've
                # re-instructed enough; a balk (written=False) re-fires the same step.
                if pend.get("shotgun"):
                    if written or (state.shotgun_retry or 0) >= _SHOTGUN_RETRY_CAP:
                        state.shotgun_idx = (state.shotgun_idx or 0) + 1
                        state.shotgun_retry = 0
                    else:
                        state.shotgun_retry = (state.shotgun_retry or 0) + 1
                sk = pend["subtask"]
                if state.subtask_done.get(sk):
                    state.stuck.pop(sk, None)
                elif written:
                    state.stuck[sk] = 0  # made progress; reset
                else:
                    state.stuck[sk] = state.stuck.get(sk, 0) + 1  # refused / vague
                if _WATCHDOG_ON:
                    # Episode-level anti-hang backstop: count turns with NO real
                    # forward motion. "Motion" = the goal field's VALUE changed (a new
                    # measurement, incl. poor→excellent) OR the pending subtask just
                    # completed. This resets on genuine progress (so a slow legit deep
                    # task is NOT cut), and only climbs when the episode is truly frozen
                    # with varied-but-empty turns the verbatim-repeat guard can't catch.
                    _adv = any(_goal_before.get(f) != state.task_state.get(f) for f in _gf) \
                        or (state.subtask_done.get(sk) and not _sk_done_before)
                    state.stall_turns = 0 if _adv else state.stall_turns + 1
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
                    state.acted.pop(pend["subtask"], None)  # L-SEQ: re-opened → re-dispatch its actions
                    state.active_subtask = None

        # ESCALATION is terminal: once transfer_to_human_agents has been dispatched
        # (supervisor or WATCHDOG), do NOT re-enter the walk — it would re-select the
        # same subtask, re-emit the same instruction, and re-escalate every turn to
        # max_steps. Emit the policy hand-off line and yield so the user simulator can
        # ###TRANSFER###/stop. (Latent before WATCHDOG: _escalate had no other heavy caller.)
        if state.task_state.get("escalated"):
            return self._say(
                "You are being transferred to a human agent. Please hold on.", state
            )

        # 2) BOOTSTRAP: we need the phone number before the schema walk can run.
        if not state.task_state.get("phone_number"):
            state.asked_phone = True
            state.pending_kind = "await_phone"
            return self._say(
                "I can help with that. Could you tell me the phone number on the account?",
                state,
            )

        # 3) RESOLVED → closing message; let the user simulator stop.
        #    Proactive goal-satisfied close: also set `resolved` when the family's
        #    success_when_all predicate already holds (a retest reached the goal
        #    state, e.g. speed_test=='excellent' / can_send_mms==true) but no
        #    VERIFY subtask formally closed it. Otherwise the agent grinds to
        #    max_steps, which the evaluator HARD-ZEROS (it never even checks the
        #    env_assertions) — turning an already-solved task into reward 0.
        #    Keys off success_when_all = the exact eval criterion, so a true
        #    predicate means the goal state is genuinely reached.
        if (_UNLATCH_ON and isinstance(message, UserMessage)
                and state.task_state.get("resolved") and state.closed_proactively):
            # The proactive close declared victory, but the customer is still talking
            # (they did not ###STOP###) → they do not agree it is fixed. Clear the
            # latch + the goal fields and re-open the verifying subtasks so the walk
            # resumes a real verification instead of repeating the closing line.
            _swa_fields = self._pred_fields(self._family_spec(state).get("success_when_all", []))
            for f in _swa_fields:
                state.task_state[f] = None
            state.task_state["resolved"] = False
            state.closed_proactively = False
            for _name in list(state.subtask_done.keys()):
                _stype = (self._subtask_by_name(state, _name) or {}).get("subtask_type", "")
                if set(self._dwa_fields(_stype)) & set(_swa_fields):
                    state.subtask_done.pop(_name, None)
                    state.acted.pop(_name, None)
            state.active_subtask = None
            _state_log({"kind": "unlatch", "task_id": tid, "cleared": _swa_fields,
                        "task_state": dict(state.task_state)})

        if not state.task_state.get("resolved") and state.repair_attempted:
            _swa = self._family_spec(state).get("success_when_all", [])
            if _swa and eval_pred(_swa, state.task_state):
                state.task_state["resolved"] = True
                state.closed_proactively = True  # L-UNLATCH: distinguish from a VERIFY-driven close
                _state_log({"kind": "close", "task_id": tid, "reason": "goal_satisfied",
                            "success_when_all": _swa, "task_state": dict(state.task_state)})
        if state.task_state.get("resolved"):
            _state_log({"kind": "stop", "task_id": tid, "reason": "resolved",
                        "task_state": dict(state.task_state)})
            return self._say(
                "Great news — that should be fully resolved now. Is there anything else I can help you with?",
                state,
            )

        # L-WATCHDOG stall: no goal-field advance for too many turns (REPLAN-thrash /
        # depth-collapse). Bound the wasted turns with an escalate rather than grinding
        # to max_steps — the task was not converging in dialogue. Efficiency ($) guard.
        if _WATCHDOG_ON and state.stall_turns >= _WATCHDOG_STALL:
            _state_log({"kind": "watchdog", "task_id": tid, "reason": "stall",
                        "turns": state.stall_turns, "task_state": dict(state.task_state)})
            return self._escalate(
                "Unable to resolve the issue within the live session after repeated attempts; "
                "escalating to a human agent.",
                state, tid,
            )

        # 4) Pick the next subtask; abandon any whose elicitation is no longer
        #    worth pursuing. SAGE on: EVPI−cost < α·max_pi (or n_a backstop);
        #    SAGE off: original consecutive no-progress ≥ 3. Either way bounds
        #    time so a refused/vague customer can't loop us to max_steps.
        # L-SHOTGUN short-circuit: in the mms repair zone, drive the fixed speculative
        # plan instead of the diagnostic walk. Bypasses give-up and done_when_all;
        # progress is tracked by state.shotgun_fired, closing via the step-3 L-CLOSE.
        shot_sub, shot_ba = (None, None)
        if self._shotgun_ready(state):
            shot_sub, shot_ba = self._shotgun_next(state)
        subtask = shot_sub if shot_sub is not None else self._active_subtask(state)
        while shot_sub is None and subtask is not None and self._should_give_up(subtask, state, tid):
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
                    state.acted.pop(target, None)  # L-SEQ: re-opened → re-dispatch its actions
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
        if shot_sub is None and subtask is None:
            # L-ELICIT: before giving up, check whether a pending on-chain repair is dead-
            # gated only by a null context-precondition the user can grant (e.g. makePayment
            # blocked on pay_allowed). If so, ask for it instead of transferring; the answer
            # unblocks the repair and the flow cascades to resolution.
            tgt = self._elicit_target(state)
            if tgt is not None:
                field, value, repair = tgt
                state.elicited = list(state.elicited or []) + [field]
                question = self._phrase_elicitation(field, repair, state)
                state.pending = {"elicit_field": field, "elicit_value": value}
                state.pending_kind = "await_elicit"
                _state_log({"kind": "elicit", "task_id": tid, "field": field, "value": value,
                            "repair": repair["name"], "question": question,
                            "task_state": dict(state.task_state)})
                return self._say(question, state)
            _state_log({"kind": "stop", "task_id": tid, "reason": "no_active_subtask",
                        "task_state": dict(state.task_state)})
            return self._say(
                "I've done what I can on this; I'll have a specialist follow up with you. Anything else?",
                state,
            )

        # 5) EXECUTE — route to the next base_action (RC1). agent-side = real tool
        #    call (restricted to that one tool); user-side = instruction text.
        if shot_ba is not None:
            # L-SHOTGUN: force base_action[0] (the mutating fix), bypassing its diagnostic
            # `when` that would otherwise fall through to the retest. The index is NOT
            # advanced here — INGEST advances it only once the fix LANDS (or the retry cap
            # is hit), so a balked instruction re-fires instead of being skipped. All plan
            # tools are user-side device actions.
            chosen, tool_name = shot_ba, shot_ba.get("tool")
            kind = "agent" if tool_name in self._agent_tool_names else "user"
            _state_log({"kind": "shotgun", "task_id": tid, "subtask": subtask["name"],
                        "tool": tool_name, "idx": state.shotgun_idx,
                        "retry": state.shotgun_retry, "task_state": dict(state.task_state)})
        else:
            kind, chosen, tool_name = self._classify(subtask, state)
        # L-CLOSE guard: record that a real device/backend repair was dispatched
        # this episode. Proactive close requires this, so a parse error that wrongly
        # sets a goal field (e.g. service_status="connected" while the device shows
        # no_service) can't close a task on which we never even tried a fix — the
        # observed service false-close ran pure diagnostics, never a mutating tool.
        if _MUT_TOOL_RE.match(tool_name or ""):
            state.repair_attempted = True
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
                _record_tokens(tid, "schema_executor", am, sim_seed=getattr(self, "_seed", None))
                if am.is_tool_call():
                    call = am.tool_calls[0]
                    state.pending = {
                        "subtask": subtask["name"],
                        "subtask_type": subtask["subtask_type"],
                        "tool_name": call.name,
                        "tool_call_id": call.id,
                    }
                    state.pending_kind = "tool"
                    if _SEQ_ON:
                        state.acted.setdefault(subtask["name"], [])
                        if call.name not in state.acted[subtask["name"]]:
                            state.acted[subtask["name"]].append(call.name)
                    _state_log({"kind": "exec", "task_id": tid, "subtask": subtask["name"],
                                "subtask_type": subtask["subtask_type"], "chosen_tool": call.name,
                                "args": call.arguments, "side": "agent",
                                "task_state": dict(state.task_state)})
                    state.messages.append(am)
                    return am, state

        # user-side (or agent tool unexpectedly unavailable → ask the customer)
        focus = self._sage_focus_hint(subtask["subtask_type"], state)
        if shot_ba is not None:
            parts = []
            # App-name grounding: the device app is literally "messaging" (lowercase). The
            # user-sim otherwise normalizes "your messaging app" → the proper noun
            # "Messages" → check_app_permissions/grant return "app not found" → perms
            # misread as missing → skip/prime defeated → balk loop (seen in 5/24 mms sims,
            # killing apn's budget and flaking the perm tasks). Pin the literal token.
            if tool_name in ("check_app_permissions", "grant_app_permission"):
                parts.append('- The app is named exactly "messaging" (all lowercase). There '
                             'is no app called "Messages" — refer to it as "messaging".')
            if (state.shotgun_retry or 0) > 0:
                # Re-instruction after a balk: push concrete, do-it-now steps so the cold
                # (diagnosis-skipped) customer actually performs the action this time.
                parts.append("- The customer hesitated last time. Give a concrete, do-it-now "
                             "instruction with the exact Settings path, and ask them to "
                             "perform it now and report the result — do not offer to skip.")
            if parts:
                focus = "\n".join(parts)
        if _REPLY_ON:
            extra = []
            # App-name grounding on the NORMAL walk (shotgun branch already pins it):
            # the device app is literally "messaging"; the user-sim rejects "Messages".
            if shot_ba is None and tool_name in ("check_app_permissions", "grant_app_permission"):
                extra.append('- The app is named exactly "messaging" (all lowercase). There is no '
                             'app called "Messages" — tell the customer to open the app named "messaging".')
            # Re-ask injection: after a balk, answer the customer's actual question
            # (give the exact name/value they asked for) instead of repeating verbatim.
            if state.last_user_reply and state.stuck.get(subtask["name"], 0) > 0:
                _r = state.last_user_reply.strip().replace("\n", " ")[:200]
                extra.append(f'- The customer just said: "{_r}". Answer their specific question or '
                             'concern directly and concretely; do NOT repeat your previous instruction verbatim.')
            if extra:
                focus = (focus + "\n" + "\n".join(extra)) if focus else "\n".join(extra)
        instr = self._phrase_instruction(subtask, chosen, state, focus=focus)
        if _WATCHDOG_ON and shot_ba is None:
            # Verbatim-deadlock guard: the same instruction re-emitted N times means
            # the customer will not act on it (e.g. demanding a value we keep not
            # giving). Stop instead of looping to max_steps.
            if instr == state.last_instr:
                state.instr_repeat += 1
            else:
                state.instr_repeat = 0
                state.last_instr = instr
            if state.instr_repeat >= _WATCHDOG_REPEAT:
                _state_log({"kind": "watchdog", "task_id": tid, "reason": "instr_repeat",
                            "count": state.instr_repeat, "subtask": subtask["name"],
                            "task_state": dict(state.task_state)})
                return self._escalate(
                    "Unable to complete troubleshooting in dialogue (repeated the same step "
                    "without progress); escalating to a human agent.",
                    state, tid,
                )
        if _SAGE_ON:
            _max_pi, _bs, _go, _det = self._sage_belief(state.family, subtask["subtask_type"], state)
            _state_log({"kind": "sage_belief", "task_id": tid, "subtask": subtask["name"],
                        "max_pi": round(_max_pi, 5), "details": _det, "focus": bool(focus)})
        state.pending = {
            "subtask": subtask["name"],
            "subtask_type": subtask["subtask_type"],
            "tool_name": tool_name,
            "shotgun": shot_ba is not None,
        }
        state.pending_kind = "user"
        if _SEQ_ON and shot_ba is None:
            state.acted.setdefault(subtask["name"], [])
            if tool_name not in state.acted[subtask["name"]]:
                state.acted[subtask["name"]].append(tool_name)
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
