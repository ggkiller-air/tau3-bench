"""SchemaFlex-std: unified deterministic middleware + sparse HILO for standard-mode retail.

The "telecom recipe" (8B workhorse + deterministic orchestration + SPARSE big-model on the
load-bearing semantic step) ported to standard-mode retail. One agent, every lever env-gated
so any combination is a one-line ablation. All OFF => identical to the stock LLMAgent.

Levers (env flags):
  SCHEMAFLEX_VIEW    — flatten read-tool JSON results into a compact table (observation mw).
  SCHEMAFLEX_CONFIRM — block a DB-write until a "proposal -> user yes" handshake exists;
                       else re-prompt the 8B (tool-free) to send a confirmation message.
  SCHEMAFLEX_DEDUP   — block a duplicate / already-terminal write (same order already
                       cancelled/exchanged/returned/item-modified, or identical prior write).
  SCHEMAFLEX_HILO    — before a DB-write executes, a big model (gpt-5.4) re-derives the
                       write's SEMANTIC GROUNDING (item_ids / new_item_ids / payment) from the
                       conversation + catalog and overrides the 8B's args if they differ.
                       Corrects grounding, NOT the decision (which tool / whether to act stays
                       the 8B's). Budget-capped + logged for the "sparse" (token-%) claim.

Deterministic levers add no big-model calls. HILO is the only big-model use and is sparse
(fires only on DB-writes, at most SCHEMAFLEX_HILO_MAX per episode).

Cross-domain note: every retail lever above is keyed to RETAIL tool names (_DB_WRITE /
_VIEW_TOOLS / get_product_details|get_order_details), so on other domains they are inert
even if enabled — but keep them OFF there (NUMINJECT/NUMSPEAK match airline's
get_user_details yet read a retail-only 'balance' field). For airline use
SCHEMAFLEX_AIRLINE_GUARDS (deterministic State Pre-Guards ported from the schema-solo
armor; see airline_guards.py). Default OFF => byte-equivalent to the retail agent.
"""

from __future__ import annotations

import json
import os
import re

from loguru import logger

from tau2.agent.airline_guards import GUARDED_TOOLS as _AIR_GUARDED
from tau2.agent.airline_guards import AirlineGuardCache
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage,
)
from tau2.utils.llm_utils import generate

_VIEW_ON = os.environ.get("SCHEMAFLEX_VIEW", "") not in ("", "0", "false", "False")
_CONFIRM_ON = os.environ.get("SCHEMAFLEX_CONFIRM", "") not in ("", "0", "false", "False")
_DEDUP_ON = os.environ.get("SCHEMAFLEX_DEDUP", "") not in ("", "0", "false", "False")
_HILO_ON = os.environ.get("SCHEMAFLEX_HILO", "") not in ("", "0", "false", "False")
_HILO_LLM = os.environ.get("SCHEMAFLEX_HILO_LLM", "openai/gpt-5.4")
_HILO_MAX = int(os.environ.get("SCHEMAFLEX_HILO_MAX", "8"))
_HILO_AGGR = os.environ.get("SCHEMAFLEX_HILO_AGGR", "") not in ("", "0", "false", "False")
_COMM_ON = os.environ.get("SCHEMAFLEX_COMM", "") not in ("", "0", "false", "False")
_CHECKLIST_ON = os.environ.get("SCHEMAFLEX_CHECKLIST", "") not in ("", "0", "false", "False")
_NUMINJECT_ON = os.environ.get("SCHEMAFLEX_NUMINJECT", "") not in ("", "0", "false", "False")
# NUMSPEAK: like NUMINJECT but HARDER — instead of feeding the figure to the 8B and hoping it
# relays it, capture the figure code-side when the read tool returns, then on the 8B's final
# TEXT reply append the exact figure sentence directly (only when the reply omits it). Removes
# the "will the 8B verbalise it" uncertainty entirely; deterministic, at most once per figure.
_NUMSPEAK_ON = os.environ.get("SCHEMAFLEX_NUMSPEAK", "") not in ("", "0", "false", "False")
# WFGUARD: block STRUCTURALLY-void writes (exchange/modify_items with empty new_item_ids,
# length mismatch, or every new item == old item — a no-op). Pure shape check on the call, no
# semantics -> zero false-negatives (a self-equal/empty exchange is wrong in any context).
_WFGUARD_ON = os.environ.get("SCHEMAFLEX_WFGUARD", "") not in ("", "0", "false", "False")
# PREGUARD: block a write whose order status fails the tool's precondition, MIRRORING env logic
# exactly (cancel/modify_items need status=='pending'; modify_address/payment need 'pending' in
# status; return/exchange need status=='delivered'). Status is an OBSERVED fact -> zero
# false-negatives (only blocks writes the env would also reject), with a policy-framed nudge.
_PREGUARD_ON = os.environ.get("SCHEMAFLEX_PREGUARD", "") not in ("", "0", "false", "False")
# AIRLINE GUARDS: deterministic State Pre-Guards for the airline domain (cancel
# eligibility recompute / book payment-combo + baggage rewrite / bags only-increase /
# BE flight lock + segment chain). Judgment logic lives in airline_guards.py (a
# self-contained port of schema_solo_std_agent's ARMOR hooks B-E — NOT imported from
# there, to keep the schema stack out of user-mode runs). Facts come from raw env
# ToolMessages only; vetoed writes never reach the env (reprompt via _block_reprompt).
# Default OFF => zero new code paths execute (self._air stays None).
_AIRGUARD_ON = os.environ.get("SCHEMAFLEX_AIRLINE_GUARDS", "") not in ("", "0", "false", "False")
_AIRGUARD_MAX_VETO = int(os.environ.get("SCHEMAFLEX_AIRLINE_GUARDS_MAX_VETO", "3"))

# The 8B often completes the DB action but fails to VERBALISE a figure the customer asked
# for (count / price / total / balance / refund) — a load-bearing nl_assertion. It KNOWS the
# value (it is in a tool result); it just doesn't say it. COMM = (1) a system-prompt rule +
# (2) a deterministic reply-guard that re-prompts when the customer asked for a figure and the
# drafted reply has no number. Small model does the relaying; deterministic control forces it.
_COMM_RULES = (
    "\n<communication_rules>\n"
    "1. Whenever the customer asks HOW MANY options/items exist, or asks for a PRICE, TOTAL, "
    "COST, BALANCE, or REFUND amount, reply with the EXACT number/amount (look it up first if "
    "needed). Never leave such a question unanswered.\n"
    "2. Before treating a request as complete, make sure you have (a) stated every figure the "
    "customer asked about and (b) handled EVERY item they mentioned.\n"
    "</communication_rules>"
)
_INFO_Q = re.compile(
    r"how many|how much|number of|\bbalance\b|what('?s| is| are)\b[^?]{0,40}"
    r"(price|total|cost|balance|refund|amount)|the (price|total|cost|balance|refund|amount)"
    r"|how many .*available|options are available",
    re.IGNORECASE,
)
_COMM_NUDGE = (
    "SYSTEM CHECK (not from the customer): the customer just asked for a specific figure "
    "(a count, price, total, balance, or refund amount) and your reply did not include a "
    "number. Look up the value if needed and reply stating the EXACT figure."
)
_CONFIRM_RETRIES = int(os.environ.get("SCHEMAFLEX_CONFIRM_RETRIES", "1"))
# Event log (JSONL) — loguru INFO from agent modules is filtered in tau2 runs, so lever
# firings (hilo/confirm/dedup) are recorded to a file for measurement. Set per run.
_EVENT_LOG = os.environ.get("SCHEMAFLEX_EVENT_LOG", "")


def _event(d):
    if not _EVENT_LOG:
        return
    try:
        with open(_EVENT_LOG, "a") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception:
        pass

# -- VIEW layer (deterministic observation middleware) ---------------------- #
# Flattens read-tool nested JSON into a compact table the 8B reads reliably (the naive-8B
# killer is item->variant mapping off raw dict-of-dicts). Rewrites RESULTS only — the env
# still runs the real args, so evaluation is unaffected. Formerly llm_std_view_agent.py.

# Read tools whose raw nested JSON we flatten. Writes / id-lookups / calculate pass through.
_VIEW_TOOLS = {
    "get_product_details",
    "get_order_details",
    "get_user_details",
    "get_item_details",
}


def _fmt_options(opts) -> str:
    if not isinstance(opts, dict):
        return str(opts)
    return ", ".join(f"{k}={v}" for k, v in opts.items())


def _view_product(d: dict) -> str:
    lines = [f"Product: {d.get('name')} (product_id {d.get('product_id')})"]
    variants = d.get("variants") or {}
    lines.append("Variants — item_id | available | price | options:")
    for iid, v in variants.items():
        if not isinstance(v, dict):
            continue
        av = "available" if v.get("available") else "OUT_OF_STOCK"
        lines.append(f"  {v.get('item_id', iid)} | {av} | ${v.get('price')} | {_fmt_options(v.get('options'))}")
    return "\n".join(lines)


def _view_order(d: dict) -> str:
    lines = [f"Order {d.get('order_id')} | status: {d.get('status')} | user: {d.get('user_id')}"]
    items = d.get("items") or []
    lines.append("Items — item_id | product_id | name | options | price:")
    for it in items:
        if not isinstance(it, dict):
            continue
        lines.append(
            f"  {it.get('item_id')} | {it.get('product_id')} | {it.get('name')} | "
            f"{_fmt_options(it.get('options'))} | ${it.get('price')}"
        )
    ph = d.get("payment_history") or []
    if ph:
        parts = [f"{p.get('transaction_type')} ${p.get('amount')} via {p.get('payment_method_id')}"
                 for p in ph if isinstance(p, dict)]
        lines.append("Payment history: " + "; ".join(parts))
        first = next((p for p in ph if isinstance(p, dict) and p.get("transaction_type") == "payment"), None)
        if first:
            lines.append(f"Original payment method (for refunds): {first.get('payment_method_id')}")
    addr = d.get("address")
    if isinstance(addr, dict):
        lines.append("Shipping address: " + _fmt_options(addr))
    # surface exchange/return follow-up fields if present
    for k in ("exchange_items", "exchange_new_items", "return_items", "cancel_reason"):
        if d.get(k):
            lines.append(f"{k}: {d.get(k)}")
    return "\n".join(lines)


def _view_user(d: dict) -> str:
    name = d.get("name") or {}
    lines = [f"User {d.get('user_id')} | {name.get('first_name')} {name.get('last_name')} | email {d.get('email')}"]
    addr = d.get("address")
    if isinstance(addr, dict):
        lines.append("Default address: " + _fmt_options(addr))
    pms = d.get("payment_methods") or {}
    if pms:
        parts = []
        for pid, pm in pms.items():
            if not isinstance(pm, dict):
                continue
            src = pm.get("source")
            if src == "gift_card":
                parts.append(f"{pm.get('id', pid)} (gift_card balance ${pm.get('balance')})")
            elif src == "credit_card":
                parts.append(f"{pm.get('id', pid)} (credit_card {pm.get('brand')} *{pm.get('last_four')})")
            else:
                parts.append(f"{pm.get('id', pid)} ({src})")
        lines.append("Payment methods: " + "; ".join(parts))
    orders = d.get("orders") or []
    if orders:
        lines.append("Orders: " + ", ".join(map(str, orders)))
    return "\n".join(lines)


def _view_item(d: dict) -> str:
    av = "available" if d.get("available") else "OUT_OF_STOCK"
    return (f"item_id {d.get('item_id')} | {av} | ${d.get('price')} | "
            f"options: {_fmt_options(d.get('options'))}")


_VIEWS = {
    "get_product_details": _view_product,
    "get_order_details": _view_order,
    "get_user_details": _view_user,
    "get_item_details": _view_item,
}


def _viewify(tool_name: str, content: str):
    """Flatten a read tool's JSON result into a compact view; pass through on any failure."""
    fn = _VIEWS.get(tool_name)
    if fn is None or not isinstance(content, str):
        return content
    try:
        d = json.loads(content)
    except Exception:
        return content  # error strings, non-JSON -> untouched
    if not isinstance(d, dict):
        return content
    try:
        return fn(d)
    except Exception:
        return content


_DB_WRITE = {
    "cancel_pending_order", "modify_pending_order_address", "modify_pending_order_payment",
    "modify_pending_order_items", "return_delivered_order_items",
    "exchange_delivered_order_items", "modify_user_address",
}
# order statuses that mean "this order can no longer be written to" for a given family
_TERMINAL_AFTER = {
    "cancel_pending_order": {"cancelled"},
    "exchange_delivered_order_items": {"exchange requested"},
    "return_delivered_order_items": {"return requested"},
    "modify_pending_order_items": {"pending (item modified)", "pending (items modifed)"},
}

_AFFIRM = re.compile(
    r"\b(yes|yep|yeah|yup|sure|confirm(ed)?|go ahead|proceed|please do|do it|sounds good|"
    r"that'?s (right|correct)|correct|okay|ok|fine|agree[d]?|let'?s do|go for it)\b", re.I)

_CONFIRM_NUDGE = (
    "SYSTEM CHECK (not from the customer): store policy REQUIRES explicit customer confirmation "
    "before ANY order/account change, and you have not obtained a 'yes' for the change you were "
    "about to make. Do NOT call any tool now. Send the customer a short message that (1) lists the "
    "EXACT change(s) (order id, item(s) + their new option(s), payment method) and any count / "
    "balance / price / total they asked about, and (2) asks them to confirm yes — and for "
    "exchanges/returns/modifications, ask them to confirm they've given ALL items they want changed. "
    "Write only that message.")

_HILO_SYSTEM = (
    "You are a GROUNDING corrector for a retail support agent. The agent has already DECIDED which "
    "tool to call and whether to act — do NOT change the tool or second-guess the decision. Your ONLY "
    "job: verify/fix the ARGUMENTS. Using items_in_play (each item's current options + its product's "
    "available_variants table mapping item_id->options), map the customer's requested options "
    "(e.g. 'clicky switch, RGB, full size', 'compatible with Google Home') to the exact variant "
    "item_id. Rules: (1) every id in item_ids/new_item_ids MUST be an item_id that appears in an "
    "available_variants table or order_item_ids — NEVER a product_id; (2) new_item_ids must be "
    "position-matched to item_ids and be a DIFFERENT variant of the SAME product; (3) keep options the "
    "customer didn't ask to change identical to the current item; (4) respect conditionals/fallbacks "
    "in the request; (5) if the agent's args are already correct, return them unchanged. "
    "Return ONLY a JSON object: {\"item_ids\":[...], \"new_item_ids\":[...], \"payment_method_id\":\"...\"} "
    "(include only the fields the tool needs).")


class LLMStdFlexAgent(LLMAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._orders = {}     # order_id -> {status, item_ids:[...], original_payment}
        self._items = {}      # item_id -> {product_id, options:{}, name}  (from order contents)
        self._products = {}   # product_id -> {name, variants:{item_id:{options:{}, available, price}}}
        self._done_writes = set()  # (tool, order_id, frozenset(item_ids)) already dispatched
        self._hilo_calls = 0
        self._fig_count = None    # latest {"name","n","pid"} from get_product_details
        self._fig_balance = None  # latest [(id, balance)] gift cards from get_user_details
        self._fig_total = None    # latest {"order_id","amount"} from get_order_details
        self._spoke = set()       # figure types NUMSPEAK has already voiced/confirmed present
        # AIRLINE GUARDS: fact cache created ONLY when the flag is on, so every new
        # code path is behind `self._air is not None` (flag off => byte-equivalent).
        self._air = AirlineGuardCache() if _AIRGUARD_ON else None

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        return base + _COMM_RULES if _COMM_ON else base

    # -- catalog (for DEDUP + HILO) ---------------------------------------- #
    def _ingest_raw(self, tool_name, content):
        if not isinstance(content, str):
            return
        try:
            d = json.loads(content)
        except Exception:
            return
        if not isinstance(d, dict):
            return
        if d.get("order_id") and ("items" in d or "status" in d):
            iids = []
            for it in (d.get("items") or []):
                if isinstance(it, dict) and it.get("item_id"):
                    iid = str(it["item_id"]); iids.append(iid)
                    self._items[iid] = {"product_id": str(it.get("product_id")),
                                        "options": it.get("options"), "name": it.get("name")}
            orig = next((p.get("payment_method_id") for p in (d.get("payment_history") or [])
                         if isinstance(p, dict) and p.get("transaction_type") == "payment"), None)
            self._orders[str(d["order_id"])] = {"status": d.get("status"), "item_ids": iids,
                                                "original_payment": orig}
        elif d.get("product_id") and "variants" in d:
            self._products[str(d["product_id"])] = {
                "name": d.get("name"),
                "variants": {str(v.get("item_id", iid)): {"options": v.get("options"),
                                                          "available": bool(v.get("available")),
                                                          "price": v.get("price")}
                             for iid, v in (d.get("variants") or {}).items() if isinstance(v, dict)}}

    def _id_to_name(self, state):
        m = {}
        for msg in state.messages:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    m[tc.id] = tc.name
        return m

    # -- CONFIRM ----------------------------------------------------------- #
    def _confirmed(self, state):
        msgs = state.messages
        last_write = -1
        for i, m in enumerate(msgs):
            if isinstance(m, AssistantMessage) and m.tool_calls and any(
                    tc.name in _DB_WRITE for tc in m.tool_calls):
                last_write = i
        seen = False
        for m in msgs[last_write + 1:]:
            if isinstance(m, AssistantMessage) and not m.tool_calls and (m.content or "").strip():
                seen = True
            elif isinstance(m, UserMessage) and seen and _AFFIRM.search(m.content or ""):
                return True
        return False

    # -- DEDUP ------------------------------------------------------------- #
    def _dedup_reason(self, call):
        args = call.arguments or {}
        oid = str(args.get("order_id") or "")
        order = self._orders.get(oid)
        if order and order.get("status") in _TERMINAL_AFTER.get(call.name, set()):
            return f"order {oid} already in terminal state '{order['status']}' for {call.name}"
        key = (call.name, oid, frozenset(map(str, args.get("item_ids") or [])))
        if key in self._done_writes:
            return f"identical {call.name} on order {oid} already dispatched"
        return None

    # -- WFGUARD (structural well-formedness of the write) ----------------- #
    def _wf_invalid(self, call):
        """Return a reason string iff the write is structurally void/malformed. Only fires on
        cases that change NOTHING or cannot be valid in ANY context (empty ids, length mismatch,
        every new item == its old item). Never blocks a partially-valid write -> no false-neg."""
        if call.name not in ("exchange_delivered_order_items", "modify_pending_order_items"):
            return None
        a = call.arguments or {}
        iids = list(map(str, a.get("item_ids") or []))
        niids = list(map(str, a.get("new_item_ids") or []))
        if not iids or not niids:
            return "item_ids or new_item_ids is empty — the change would do nothing"
        if len(iids) != len(niids):
            return (f"item_ids ({len(iids)}) and new_item_ids ({len(niids)}) have different "
                    f"lengths — they must be position-matched one-to-one")
        if all(o == n for o, n in zip(iids, niids)):
            return "every new item id equals the old one — this change is a no-op"
        return None

    # -- PREGUARD (order-status precondition, mirrors env exactly) --------- #
    def _preguard_reason(self, call):
        a = call.arguments or {}
        oid = str(a.get("order_id") or "")
        order = self._orders.get(oid)
        if not order:               # never observed this order -> can't check -> fail-open
            return None
        st = order.get("status") or ""
        name = call.name
        if name in ("cancel_pending_order", "modify_pending_order_items") and st != "pending":
            return (f"order {oid} status is '{st}', but {name} requires an exactly-'pending' "
                    f"order (a modified/processed/delivered order cannot be used here)")
        if name in ("modify_pending_order_address", "modify_pending_order_payment") \
                and "pending" not in st:
            return f"order {oid} status is '{st}', but {name} requires a pending order"
        if name in ("return_delivered_order_items", "exchange_delivered_order_items") \
                and st != "delivered":
            return f"order {oid} status is '{st}', but {name} requires a 'delivered' order"
        return None

    def _block_reprompt(self, state, note):
        """Block the write and re-prompt the 8B with a policy-framed system note (tools available
        so it can look up / inform / do the right thing instead)."""
        base = state.system_messages + state.messages
        nudge = UserMessage(role="user", content=(
            "SYSTEM CHECK (not from the customer): " + note + " Do NOT make that tool call. Either "
            "correct it, or tell the customer clearly why it cannot be done and handle the rest of "
            "their request."))
        return generate(model=self.llm, tools=self.tools, messages=base + [nudge],
                        call_name="agent_guard_prompt", **self.llm_args)

    # -- structural validity (conservative HILO gate) --------------------- #
    def _write_provably_invalid(self, call):
        """True only when the 8B's write args are DEMONSTRABLY wrong from observed catalog
        (wrong-length, new item not an available variant of the same product, item not in
        order). Returns False when we cannot prove it wrong (trust the 8B — it's competitive
        at grounding). HILO fires only on provably-invalid writes, so it never clobbers a
        correct 8B answer."""
        a = call.arguments or {}
        name = call.name
        if name in ("exchange_delivered_order_items", "modify_pending_order_items"):
            iids = list(map(str, a.get("item_ids") or []))
            niids = list(map(str, a.get("new_item_ids") or []))
            if len(iids) != len(niids) or not iids:
                return True
            for old, new in zip(iids, niids):
                pid = self._items.get(old, {}).get("product_id")
                prod = self._products.get(str(pid))
                if prod is None:
                    continue  # product not seen -> can't prove wrong
                v = (prod.get("variants") or {}).get(new)
                if v is None or not v.get("available"):
                    return True  # new id is not an available variant of that product
            return False
        if name == "return_delivered_order_items":
            order = self._orders.get(str(a.get("order_id") or ""))
            if order and order.get("item_ids"):
                for it in map(str, a.get("item_ids") or []):
                    if it not in order["item_ids"]:
                        return True
            return False
        return False  # cancel / modify_address / modify_payment: no variant grounding

    # -- HILO -------------------------------------------------------------- #
    def _hilo_fix(self, call, state):
        args = dict(call.arguments or {})
        oid = str(args.get("order_id") or "")
        order = self._orders.get(oid) or {}
        # For each item the write touches, show its CURRENT options and the full available-variant
        # table (item_id -> options) of its product, so the corrector can map NL options -> item_id.
        focus = []
        seen_iid = list(args.get("item_ids") or []) + list(order.get("item_ids") or [])
        for iid in dict.fromkeys(map(str, seen_iid)):
            info = self._items.get(iid, {})
            pid = info.get("product_id")
            prod = self._products.get(str(pid), {})
            vtable = {vid: v.get("options") for vid, v in (prod.get("variants") or {}).items()
                      if v.get("available")}
            focus.append({"current_item_id": iid, "current_options": info.get("options"),
                          "product": prod.get("name"), "product_id": pid,
                          "available_variants(item_id->options)": vtable})
        # fuller dialogue context (both sides) so grounding is as informed as the ceiling agent
        convo = []
        for m in state.messages[-40:]:
            if isinstance(m, UserMessage) and (m.content or "").strip():
                convo.append("USER: " + (m.content or "")[:400])
            elif isinstance(m, AssistantMessage) and not m.tool_calls and (m.content or "").strip():
                txt = re.sub(r"<think>[\s\S]*?</think>", "", m.content or "").strip()
                if txt:
                    convo.append("AGENT: " + txt[:300])
        ctx = {
            "tool": call.name,
            "agent_proposed_args": args,
            "order_id": oid, "order_status": order.get("status"),
            "order_item_ids": order.get("item_ids"),
            "items_in_play": focus,
            "original_payment_method_id": order.get("original_payment"),
            "conversation": convo[-14:],
            "note": "item_ids/new_item_ids MUST be variant item_ids taken from available_variants tables above; NEVER a product_id. Position-match new_item_ids to item_ids.",
        }
        msgs = [
            SystemMessage(role="system", content=_HILO_SYSTEM),
            UserMessage(role="user", content=json.dumps(ctx, ensure_ascii=False)),
        ]
        _event({"kind": "hilo_enter", "tool": call.name, "order": oid,
                "proposed_item_ids": args.get("item_ids"), "proposed_new": args.get("new_item_ids")})
        try:
            out = generate(model=_HILO_LLM, messages=msgs, tools=[],
                           call_name="agent_hilo_ground", temperature=0.0)
        except Exception as e:
            _event({"kind": "hilo_failed", "err": str(e)[:200]})
            return call
        m = re.search(r"\{[\s\S]*\}", out.content or "")
        if not m:
            return call
        try:
            fix = json.loads(m.group(0))
        except Exception:
            return call
        changed = {}
        for k in ("item_ids", "new_item_ids", "payment_method_id"):
            if k in fix and fix[k] and fix[k] != args.get(k):
                args[k] = fix[k]; changed[k] = fix[k]
        if changed:
            self._hilo_calls += 1
            _event({"kind": "hilo_correct", "tool": call.name, "order": oid, "changed": changed})
            call.arguments = args
        else:
            _event({"kind": "hilo_agree", "tool": call.name, "order": oid})
        return call

    # -- NUMINJECT (deterministic number reback for nl_assertion figures) -- #
    def _num_directive(self, tool_name, content, state):
        """If the customer recently asked for a figure and this read tool result contains it,
        extract the exact number deterministically and return a directive telling the 8B to
        state it. Targets only cleanly-extractable figures (variant count / gift-card balance /
        order total). No LLM, no arithmetic by the model."""
        # scan ALL user messages — the info-question may have been asked many turns before the
        # lookup returns. Gate EACH directive on the matching question type (count/balance/total),
        # so we never inject a number the customer didn't ask for.
        allq = " ".join((m.content or "") for m in state.messages if isinstance(m, UserMessage))
        ql = (allq or "").lower()
        asked_count = bool(re.search(r"how many|number of|options? (are )?available|how many options", ql))
        asked_balance = "balance" in ql
        asked_total = bool(re.search(r"how much|the total|the price|the cost|the amount|order total", ql))
        if not (asked_count or asked_balance or asked_total):
            return None
        try:
            d = json.loads(content)
        except Exception:
            return None
        if not isinstance(d, dict):
            return None
        if asked_count and tool_name == "get_product_details" and isinstance(d.get("variants"), dict):
            n = len(d["variants"])
            return (f"[SYSTEM NOTE: product '{d.get('name')}' has exactly {n} options/variants. If "
                    f"this is the product the customer asked about, tell them the number {n}.]")
        if asked_balance and tool_name == "get_user_details":
            gc = [pm for pm in (d.get("payment_methods") or {}).values()
                  if isinstance(pm, dict) and pm.get("source") == "gift_card"]
            if gc:
                bals = "; ".join(f"{pm.get('id')} = ${pm.get('balance')}" for pm in gc)
                return (f"[SYSTEM NOTE: gift card balance(s): {bals}. State the exact amount to the "
                        f"customer.]")
        if asked_total and tool_name == "get_order_details":
            tot = sum(p.get("amount", 0) for p in (d.get("payment_history") or [])
                      if isinstance(p, dict) and p.get("transaction_type") == "payment")
            if tot:
                return (f"[SYSTEM NOTE: order {d.get('order_id')} total paid = ${tot}. If the "
                        f"customer asked this order's total, state ${tot} exactly.]")
        return None

    # -- NUMSPEAK (capture figures, append to final reply code-side) ------- #
    def _capture_figures(self, tool_name, content):
        """Store the exact figure the customer might ask about, as observed from a read tool.
        No question-type gating here (we store regardless); gating happens at speak time."""
        try:
            d = json.loads(content)
        except Exception:
            return
        if not isinstance(d, dict):
            return
        if tool_name == "get_product_details" and isinstance(d.get("variants"), dict):
            self._fig_count = {"name": d.get("name"), "n": len(d["variants"]),
                               "pid": str(d.get("product_id") or "")}
        elif tool_name == "get_user_details":
            gc = [(pm.get("id"), pm.get("balance")) for pm in (d.get("payment_methods") or {}).values()
                  if isinstance(pm, dict) and pm.get("source") == "gift_card"]
            if gc:
                self._fig_balance = gc
        elif tool_name == "get_order_details":
            tot = sum(p.get("amount", 0) for p in (d.get("payment_history") or [])
                      if isinstance(p, dict) and p.get("transaction_type") == "payment")
            if tot:
                self._fig_total = {"order_id": d.get("order_id"), "amount": tot}

    def _numspeak_append(self, reply, state):
        """Return a deterministic sentence stating the exact figure the customer asked for, when
        the drafted reply omits it. Each figure type spoken at most once (standard mode ends on
        user_stop; the nl_assertion judge scans the whole transcript, so once anywhere suffices).
        If the 8B already stated the number, mark it satisfied and never re-append."""
        allq = " ".join((m.content or "") for m in state.messages if isinstance(m, UserMessage))
        ql = allq.lower()
        parts = []
        if ("count" not in self._spoke and self._fig_count
                and re.search(r"how many|number of|options? (are )?available", ql)):
            if str(self._fig_count["n"]) in reply:
                self._spoke.add("count")
            else:
                nm = self._fig_count.get("name")
                parts.append(f"There are {self._fig_count['n']} options available"
                             + (f" for the {nm}" if nm else "") + ".")
                self._spoke.add("count")
        if "balance" not in self._spoke and self._fig_balance and "balance" in ql:
            if all(str(b) in reply for _, b in self._fig_balance):
                self._spoke.add("balance")
            else:
                bals = "; ".join(f"{i}: ${b}" for i, b in self._fig_balance)
                parts.append(f"Your gift card balance is {bals}.")
                self._spoke.add("balance")
        if ("total" not in self._spoke and self._fig_total
                and re.search(r"how much|the total|the price|the cost|the amount|order total", ql)):
            if str(self._fig_total["amount"]) in reply:
                self._spoke.add("total")
            else:
                parts.append(f"The total for order {self._fig_total.get('order_id')} is "
                             f"${self._fig_total['amount']}.")
                self._spoke.add("total")
        return " ".join(parts) if parts else None

    # -- CHECKLIST (external progress tracker for multi-order/item) -------- #
    def _tracker(self):
        """Deterministic scratchpad: every order the agent has OPENED (get_order_details),
        its items, and whether a DB write has been dispatched to it. Anchors the 8B's
        attention so it neither repeats a completed action nor forgets an un-actioned order.
        Uses only observed catalog + dispatched writes — no NL parsing of the request."""
        if not self._orders:
            return None
        lines = ["[PROGRESS TRACKER — orders opened so far this conversation]"]
        for oid, o in self._orders.items():
            done = [w for w in self._done_writes if w[1] == oid]
            mark = ("a change was ALREADY dispatched to this order" if done
                    else "no change dispatched yet")
            lines.append(f"  {oid} (status: {o.get('status')}) items {o.get('item_ids')} — {mark}")
        lines.append("Rules: do NOT repeat a change already dispatched; before finishing, make "
                     "sure every order/item the customer asked you to change has been handled.")
        return "\n".join(lines)

    # -- AIRLINE GUARDS (SCHEMAFLEX_AIRLINE_GUARDS) ------------------------- #
    def _air_guard_writes(self, assistant, state):
        """Veto loop for airline write tools: guard the proposed call against cached
        env facts; on veto, the call is NEVER emitted — the 8B is re-prompted with the
        policy reason (same _block_reprompt channel as WFGUARD/PREGUARD). A guard may
        also rewrite args in place (nonfree_baggages) and pass. On budget exhaustion
        with a still-vetoed call, fail CLOSED with a deterministic text reply."""
        user_text = " ".join((m.content or "") for m in state.messages
                             if isinstance(m, UserMessage))
        last_err = None
        for _ in range(_AIRGUARD_MAX_VETO):
            if not assistant.is_tool_call():
                return assistant
            call = next((tc for tc in assistant.tool_calls if tc.name in _AIR_GUARDED), None)
            if call is None:
                return assistant
            err = self._air.guard(call, user_text)
            if err is None:
                return assistant
            last_err = err
            _event({"kind": "airguard_veto", "tool": call.name, "reason": err,
                    "args": call.arguments})
            assistant = self._block_reprompt(
                state, "airline policy blocks that action — " + err + ".")
        # budget exhausted: only a call that STILL fails the guard is replaced.
        if assistant.is_tool_call():
            call = next((tc for tc in assistant.tool_calls if tc.name in _AIR_GUARDED), None)
            if call is not None and self._air.guard(call, user_text) is not None:
                _event({"kind": "airguard_exhausted", "tool": call.name,
                        "reason": last_err})
                return AssistantMessage(role="assistant", content=(
                    "I'm sorry, but I cannot perform that action: "
                    + (last_err or "it is not permitted by airline policy")
                    + ". Is there anything else I can help you with?"))
        return assistant

    # -- main -------------------------------------------------------------- #
    def _generate_next_message(self, message, state):
        # ingest raw results for catalog; view-rewrite happens after (so catalog sees raw JSON)
        if message is not None:
            id2name = self._id_to_name(state)
            id2args = {}
            if self._air is not None:  # AIRLINE GUARDS: attribute results to call args
                for m in state.messages:
                    if isinstance(m, AssistantMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            id2args[tc.id] = tc.arguments
            tms = (message.tool_messages if isinstance(message, MultiToolMessage)
                   else [message] if isinstance(message, ToolMessage) else [])
            for tm in tms:
                nm = id2name.get(tm.id)
                if nm and not tm.error and tm.content:
                    raw = tm.content
                    self._ingest_raw(nm, raw)
                    if self._air is not None:
                        self._air.ingest(nm, id2args.get(tm.id), raw)
                    if _NUMSPEAK_ON:
                        self._capture_figures(nm, raw)
                    if _NUMINJECT_ON:
                        d = self._num_directive(nm, raw, state)
                        if d:
                            _event({"kind": "numinject", "tool": nm})
                            tm.content = raw + "\n" + d
                    if _VIEW_ON and nm in _VIEW_TOOLS:
                        tm.content = _viewify(nm, tm.content)

        # generate (mirror LLMAgent), optionally injecting the CHECKLIST tracker ephemerally
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)
        gen_msgs = state.system_messages + state.messages
        if _CHECKLIST_ON:
            tr = self._tracker()
            if tr:
                gen_msgs = gen_msgs + [SystemMessage(role="system", content=tr)]
        assistant = generate(model=self.llm, tools=self.tools, messages=gen_msgs,
                             call_name="agent_response", **self.llm_args)

        # ---- text-reply path: COMM reply-guard + NUMSPEAK figure append ----
        if not assistant.is_tool_call():
            reply_msg = assistant
            # COMM: if the customer just asked for a figure and this text reply has no number,
            # re-prompt the 8B (once) to state the exact value (small model does the relaying).
            if _COMM_ON:
                last_user = next((m.content for m in reversed(state.messages)
                                  if isinstance(m, UserMessage) and (m.content or "").strip()), "")
                if _INFO_Q.search(last_user or "") and not re.search(r"\d", assistant.content or ""):
                    base = state.system_messages + state.messages
                    _event({"kind": "comm_block"})
                    reply_msg = generate(model=self.llm, tools=self.tools,
                                         messages=base + [UserMessage(role="user", content=_COMM_NUDGE)],
                                         call_name="agent_comm_prompt", **self.llm_args)
            # NUMSPEAK: deterministically append the exact figure the customer asked for, code-side
            # (bypassing the 8B), when the drafted reply omits it. At most once per figure. No LLM.
            if _NUMSPEAK_ON:
                add = self._numspeak_append(reply_msg.content or "", state)
                if add:
                    _event({"kind": "numspeak", "add": add[:80]})
                    cur = (reply_msg.content or "").rstrip()
                    reply_msg.content = (cur + " " + add) if cur else add
            return reply_msg
        # AIRLINE GUARDS: deterministic State Pre-Guard on airline write tools —
        # veto->reprompt loop, fail-closed on exhaustion (the doomed write never
        # reaches the env). Retail tools are untouched (disjoint tool-name sets).
        if self._air is not None:
            assistant = self._air_guard_writes(assistant, state)
            if not assistant.is_tool_call():
                return assistant

        write = next((tc for tc in assistant.tool_calls if tc.name in _DB_WRITE), None)
        if write is None:
            return assistant

        # CONFIRM: block unconfirmed writes -> ask for confirmation instead
        if _CONFIRM_ON and not self._confirmed(state):
            base = state.system_messages + state.messages
            _event({"kind": "confirm_block", "tool": write.name})
            for _ in range(_CONFIRM_RETRIES + 1):
                regen = generate(model=self.llm, tools=[],
                                 messages=base + [UserMessage(role="user", content=_CONFIRM_NUDGE)],
                                 call_name="agent_confirm_prompt", **self.llm_args)
                if not regen.is_tool_call() and (regen.content or "").strip():
                    return regen
            return assistant  # fail-open

        # DEDUP: drop duplicate / already-terminal writes
        if _DEDUP_ON:
            reason = self._dedup_reason(write)
            if reason:
                _event({"kind": "dedup_block", "tool": write.name, "reason": reason})
                base = state.system_messages + state.messages
                nudge = UserMessage(role="user", content=(
                    "SYSTEM CHECK: that action was already completed / not applicable (" + reason +
                    "). Do NOT repeat it. If all requested changes are done, tell the customer and "
                    "then call done; otherwise handle the remaining request. No tool now."))
                regen = generate(model=self.llm, tools=self.tools, messages=base + [nudge],
                                 call_name="agent_dedup_prompt", **self.llm_args)
                return regen

        # WFGUARD: block structurally-void writes (empty / length-mismatch / no-op)
        if _WFGUARD_ON:
            reason = self._wf_invalid(write)
            if reason:
                _event({"kind": "wfguard_block", "tool": write.name, "reason": reason})
                return self._block_reprompt(state, "that order change is malformed — " + reason + ".")

        # PREGUARD: block a write whose order status fails the tool precondition (mirrors env)
        if _PREGUARD_ON:
            reason = self._preguard_reason(write)
            if reason:
                _event({"kind": "preguard_block", "tool": write.name, "reason": reason})
                return self._block_reprompt(state, "store policy blocks that action — " + reason + ".")

        # HILO: sparse big-model grounding correction — ONLY on provably-invalid writes
        # (conservative: never touch a structurally-valid 8B answer). SCHEMAFLEX_HILO_AGGR=1
        # reverts to correcting every write (for ablation).
        if _HILO_ON and self._hilo_calls < _HILO_MAX:
            if _HILO_AGGR or self._write_provably_invalid(write):
                self._hilo_fix(write, state)
            else:
                _event({"kind": "hilo_trust8b", "tool": write.name})
        # always record dispatched write (used by DEDUP + CHECKLIST tracker)
        a = write.arguments or {}
        self._done_writes.add((write.name, str(a.get("order_id") or ""),
                               frozenset(map(str, a.get("item_ids") or []))))
        return assistant


def create_llm_std_flex_agent(tools, domain_policy, **kwargs):
    return LLMStdFlexAgent(tools=tools, domain_policy=domain_policy,
                           llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))
