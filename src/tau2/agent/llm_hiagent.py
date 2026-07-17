"""SLM-HiAgent: training-free hierarchical working-memory baseline (HiAgent, ACL 2025).

Faithful port of HiAgent (Hu et al., 2025, arXiv:2408.09559, ACL 2025 Long) to tau2
turn-based dialogue. The agent manages context at SUBGOAL granularity: the running
episode is segmented into subgoal segments; when the model (a same-backbone side call)
declares a subgoal complete and opens the next one, the closed segment's full
action-observation detail is replaced in the working context by the subgoal line plus a
one-line summarized observation with a met/not-met verdict (paper Appendix C.1 prompt).
A `retrieve_subgoal_trajectory` pseudo-tool (paper §3.4) lets the actor restore a hidden
segment's full detail; the call is intercepted agent-side, never reaching the environment.

Observed on qwen3-8b (kept as honest baseline behavior, not patched): subgoal-level
compression hides the earlier turns that demonstrate the dual-control protocol (device
actions are performed by the USER on instruction), after which the 8B actor drifts into
calling nonexistent agent-side device tools and re-issuing already-answered queries. A
granularity anchor in the controller prompt (mirroring the original's in-context example
role) reduces over-segmentation (67%->40% switch rate) but the protocol-drift failure
mode remains — the FC baseline passes the same easy task the HiAgent port fails.

Structure notes (deviations from the original are documented in the port spec / paper
reproducibility statement — the load-bearing ones):
- Subgoal emission/completion judgment runs as a separate same-model controller call per
  turn instead of being fused into the action completion (native function calling on the
  actor cannot reliably co-emit free text + a tool call).
- Summaries are computed once at segment close and cached (paper semantics; the official
  repo lazily re-summarizes every turn).
- `state.messages` stays the full uncompressed transcript; compression happens only in
  the working list passed to generate() (non-destructive: the transcript is never spliced).

Env flags (read once at import):
  HIAGENT_RETRIEVE      — 1 (default) include the retrieval module; 0 = paper's "w/o TR" ablation.
  HIAGENT_MAX_RETRIEVE  — max retrieval re-prompts per turn (default 2; original recursion unbounded).
  HIAGENT_NO_SUMM       — 1 = skip LLM summarization, always use last observation ("w/o OS" ablation).
  HIAGENT_TOKEN_LOG     — sidecar JSON for token accounting (controller/summarize side calls
                          never land in results.json, so they must be logged out-of-band).
  HIAGENT_DEBUG         — path to a JSONL event log (controller decisions, closes, retrievals).
"""

import json
import os
import re
import uuid
from typing import List, Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

_RETRIEVE_ON = os.environ.get("HIAGENT_RETRIEVE", "1") not in ("", "0", "false", "False")
_MAX_RETRIEVE = int(os.environ.get("HIAGENT_MAX_RETRIEVE", "2"))
_NO_SUMM = os.environ.get("HIAGENT_NO_SUMM", "") not in ("", "0", "false", "False")

_FALLBACK_SUBGOAL = "Resolve the customer's reported issue"
_RETRIEVE_TOOL_NAME = "retrieve_subgoal_trajectory"
_MSG_TRUNC = 2000

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


# --------------------------------------------------------------------------- #
# Prompts (P1 addendum / P2 controller / P3 summarizer — see port spec)
# --------------------------------------------------------------------------- #
CONTEXT_NOTE = """
Context note: To keep the context short, the conversation history for previously
completed subgoals has been hidden and replaced by the subgoal plus a one-line
summarized observation. The current subgoal and its full recent history are shown
in full. Current subgoal: {current_subgoal}.
""".strip()

RETRIEVE_NOTE = """
If the detailed hidden trajectory of a previously completed subgoal is crucial now,
call the tool `retrieve_subgoal_trajectory(subgoal_ids=[...])` to restore its full
detail before acting.
""".strip()

CONTROLLER_PROMPT = """
You manage the subgoals of a customer-service agent. A subgoal is a milestone goal that
the agent needs to complete in order to resolve the customer's issue according to policy.
You will see the working memory: summaries of previously satisfied subgoals, followed by
the full detail of the current subgoal. If there is no current subgoal, or you believe
the previous subgoal has been completed (based on past actions and observations), output
the next subgoal to be completed as a single line in the format `Subgoal: {subgoal}`.
Otherwise output exactly `CONTINUE`. A subgoal must be one line of text with no newline
characters. Do not output anything else.

Granularity: a subgoal is a MILESTONE spanning several actions, not a single step.
For example, for a customer reporting a malfunctioning device, appropriate subgoals are
milestones like "Identify the customer's account and the affected line", "Diagnose the
root cause of the malfunction", "Apply the fix and verify the issue is resolved" — each
covering multiple actions and observations. Do NOT declare a subgoal complete after every
single action; declare it complete only when its milestone outcome is clearly achieved,
and prefer `CONTINUE` when in doubt.
""".strip()

# Paper Appendix C.1, zero-shot, plus one literals-preservation sentence for the
# dialogue setting (deviation D4).
SUMMARIZER_PROMPT = """
You are an advanced AI system tasked with summarizing and analyzing a series of
action-observation pairs (trajectories) and determining whether a specific subgoal has
been met. Your goal is to create a summary that captures all essential information,
decisions, and outcomes from the given trajectories, and indicate whether the subgoal
has been met based on the summarized observations. If there are no valid actions taken,
you need to analyze the reason. Preserve exact literals that may be needed later (phone
numbers, IDs, tool argument values, error strings, settings names) — never paraphrase a
literal.

### Instructions:
1. Provide a summarized observation related to the subgoal in a concise manner.
2. Determine whether the subgoal has been met.
3. Do not output anything except the summary and whether the subgoal is met. Your output
   should be only one line. Do not output things like '##Summary'.
""".strip()


class _RetrieveTool:
    """Agent-side pseudo-tool: only .name/.openai_schema are read by generate();
    the call is intercepted before the orchestrator, never executed anywhere."""

    name = _RETRIEVE_TOOL_NAME

    @property
    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": _RETRIEVE_TOOL_NAME,
                "description": (
                    "Restore the full hidden action-observation detail of previously "
                    "completed subgoals into your context. Use judiciously."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subgoal_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Ids of previously completed subgoals to restore",
                        }
                    },
                    "required": ["subgoal_ids"],
                },
            },
        }


_RETRIEVE_TOOL = _RetrieveTool()


# --------------------------------------------------------------------------- #
# Sidecar token accounting (side calls never land in results.json; log them here)
# --------------------------------------------------------------------------- #
_HIAGENT_TOKENS: dict = {}


def _record_tokens(agent_key: str, role: str, msg, sim_seed=None) -> None:
    u = getattr(msg, "usage", None)
    if isinstance(u, dict):
        p, c = (u.get("prompt_tokens") or 0), (u.get("completion_tokens") or 0)
    elif u is not None:
        p, c = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
    else:
        return
    rec = _HIAGENT_TOKENS.setdefault(agent_key, {
        "task_id": None, "seed": sim_seed,
        "main": 0, "controller": 0, "summarize": 0,
        "prompt": 0, "completion": 0, "calls": 0,
    })
    rec[role] = rec.get(role, 0) + p + c
    rec["prompt"] += p
    rec["completion"] += c
    rec["calls"] += 1
    path = os.environ.get("HIAGENT_TOKEN_LOG")
    if path:
        try:
            json.dump(_HIAGENT_TOKENS, open(path, "w"))
        except Exception:
            pass


def _debug(event: dict) -> None:
    path = os.environ.get("HIAGENT_DEBUG")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Rendering (plain-text blocks for the summarizer)
# --------------------------------------------------------------------------- #
def _render_messages(messages: list) -> list[str]:
    lines = []
    for m in messages:
        if isinstance(m, ToolMessage):
            lines.append(f"TOOL -> {(m.content or '')[:_MSG_TRUNC]}")
        elif isinstance(m, AssistantMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    lines.append(f"AGENT called tool {tc.name}({tc.arguments})"[:_MSG_TRUNC])
            if m.content:
                lines.append(f"AGENT: {_strip_think(m.content)[:_MSG_TRUNC]}")
        else:
            lines.append(f"USER: {(m.content or '')[:_MSG_TRUNC]}")
    return lines


class LLMHiAgent(LLMAgent):
    """LLMAgent + HiAgent subgoal-granularity working-memory management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hiagent_key = uuid.uuid4().hex
        self._seed = None
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._prefix: list = []          # everything before the first subgoal, verbatim forever
        self._segments: list[dict] = []  # {id, subgoal, messages, summary}; last = current
        self._retrieved_ids: set[int] = set()

    def set_seed(self, seed: int):
        self._seed = seed
        super().set_seed(seed)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        self._reset_episode()
        state = super().get_init_state(message_history)
        self._prefix = list(state.messages)  # resumed history stays verbatim
        return state

    # -- side-call plumbing -------------------------------------------------- #
    def _side_gen_kwargs(self) -> dict:
        """llm_args for controller/summarizer calls: suppress Qwen3 thinking;
        the MAIN calls stay identical to the llm_agent
        baseline for actor comparability."""
        kw = dict(self.llm_args or {})
        if "qwen" in (self.llm or "").lower():
            eb = dict(kw.get("extra_body") or {})
            ctk = dict(eb.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            eb["chat_template_kwargs"] = ctk
            kw["extra_body"] = eb
        return kw

    def _working_memory_text(self) -> str:
        """Render closed-segment summaries + current segment detail for the controller."""
        lines = []
        for seg in self._segments[:-1]:
            lines.append(f"{seg['id']} Subgoal: {seg['subgoal']}")
            lines.append(f"Observation: {seg['summary']}")
        cur = self._segments[-1] if self._segments else None
        if cur is not None:
            lines.append(f"Current subgoal: {cur['subgoal']}")
            lines.extend(_render_messages(cur["messages"]))
        else:
            lines.extend(_render_messages(self._prefix))
            lines.append("Current subgoal: NONE")
        return "\n".join(lines)

    def _call_controller(self) -> Optional[str]:
        """P2: returns a new subgoal string, or None for CONTINUE."""
        try:
            msg = generate(
                model=self.llm,
                messages=[
                    SystemMessage(role="system", content=CONTROLLER_PROMPT),
                    UserMessage(role="user", content=self._working_memory_text()),
                ],
                call_name="hiagent_controller",
                **self._side_gen_kwargs(),
            )
        except Exception as e:
            logger.warning(f"HiAgent controller call failed, CONTINUE: {e}")
            return None
        _record_tokens(self._hiagent_key, "controller", msg, self._seed)
        text = _strip_think(msg.content or "").strip()
        for line in text.splitlines():
            if "Subgoal:" in line:
                sub = line.split("Subgoal:", 1)[1].strip()
                if sub:
                    _debug({"kind": "controller", "decision": "new_subgoal", "subgoal": sub})
                    return sub
        _debug({"kind": "controller", "decision": "continue", "raw": text[:120]})
        return None

    def _summarize_segment(self, seg: dict) -> str:
        """P3: one-line summary + met/not-met verdict; falls back to last observation."""
        def last_obs() -> str:
            for m in reversed(seg["messages"]):
                if isinstance(m, (ToolMessage, UserMessage)) and m.content:
                    return m.content[:_MSG_TRUNC]
            return "(no observation)"

        if _NO_SUMM:
            return last_obs()
        rendered = "\n".join(_render_messages(seg["messages"]))
        try:
            msg = generate(
                model=self.llm,
                messages=[
                    SystemMessage(role="system", content=SUMMARIZER_PROMPT),
                    UserMessage(
                        role="user",
                        content=f"##Trajectory\n{rendered}\n##Subgoal:\n{seg['subgoal']}\n###Output:",
                    ),
                ],
                call_name="hiagent_summarize",
                **self._side_gen_kwargs(),
            )
        except Exception as e:
            logger.warning(f"HiAgent summarizer failed, using last observation: {e}")
            return last_obs()
        _record_tokens(self._hiagent_key, "summarize", msg, self._seed)
        summary = " ".join(_strip_think(msg.content or "").split())
        return summary or last_obs()

    # -- working-context builder --------------------------------------------- #
    def _build_working(self, state: LLMAgentState) -> list:
        cur = self._segments[-1]
        addendum = CONTEXT_NOTE.format(current_subgoal=cur["subgoal"])
        if _RETRIEVE_ON and len(self._segments) > 1:
            addendum += "\n" + RETRIEVE_NOTE
        system = [
            SystemMessage(
                role="system",
                content=state.system_messages[0].content + "\n\n" + addendum,
            )
        ]
        working: list = list(self._prefix)
        for seg in self._segments[:-1]:
            if seg["id"] in self._retrieved_ids:
                working.extend(seg["messages"])
            else:
                working.append(
                    UserMessage(
                        role="user",
                        content=(
                            f"[Hidden subgoal {seg['id']}] Subgoal: {seg['subgoal']}\n"
                            f"Observation (summary): {seg['summary']}"
                        ),
                    )
                )
        working.extend(cur["messages"])
        # debug guard: no orphaned ToolMessage in the working window
        seen_ids = set()
        for m in working:
            if isinstance(m, AssistantMessage) and m.tool_calls:
                seen_ids.update(tc.id for tc in m.tool_calls)
            elif isinstance(m, ToolMessage) and m.id not in seen_ids:
                logger.warning(f"HiAgent working context has orphan ToolMessage {m.id}")
                _debug({"kind": "orphan_tool_message", "id": m.id})
        return system + working

    # -- main turn loop ------------------------------------------------------ #
    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        # 1. ingest into the true transcript AND the segment structure
        incoming = (
            list(message.tool_messages)
            if isinstance(message, MultiToolMessage)
            else [message]
        )
        state.messages.extend(incoming)
        if self._segments:
            self._segments[-1]["messages"].extend(incoming)
        else:
            self._prefix.extend(incoming)

        # 2. controller step
        if not self._segments:
            sub = self._call_controller() or self._call_controller() or _FALLBACK_SUBGOAL
            self._segments.append({"id": 1, "subgoal": sub, "messages": [], "summary": None})
            _debug({"kind": "open_segment", "id": 1, "subgoal": sub})
        else:
            cur = self._segments[-1]
            has_action = any(isinstance(m, AssistantMessage) for m in cur["messages"])
            if has_action:  # forced-CONTINUE guard: a subgoal needs >=1 action first
                sub = self._call_controller()
                if sub:
                    cur["summary"] = self._summarize_segment(cur)
                    _debug({"kind": "close_segment", "id": cur["id"],
                            "summary": cur["summary"][:200]})
                    self._retrieved_ids.clear()
                    self._segments.append({
                        "id": cur["id"] + 1, "subgoal": sub, "messages": [], "summary": None,
                    })
                    _debug({"kind": "open_segment", "id": cur["id"] + 1, "subgoal": sub})

        # 3-5. build working context, generate, intercept retrievals
        tools = list(self.tools)
        if _RETRIEVE_ON and len(self._segments) > 1:
            tools.append(_RETRIEVE_TOOL)
        retrievals_left = _MAX_RETRIEVE
        while True:
            working = self._build_working(state)
            _debug({"kind": "working_size", "n_messages": len(working)})
            assistant_message = generate(
                model=self.llm,
                tools=tools,
                messages=working,
                call_name="agent_response",
                **self.llm_args,
            )
            _record_tokens(self._hiagent_key, "main", assistant_message, self._seed)
            retrieve_calls = [
                tc for tc in (assistant_message.tool_calls or [])
                if tc.name == _RETRIEVE_TOOL_NAME
            ]
            if not retrieve_calls:
                break
            valid = {s["id"] for s in self._segments[:-1]}
            asked = []
            for tc in retrieve_calls:
                ids = tc.arguments.get("subgoal_ids") if isinstance(tc.arguments, dict) else None
                for i in ids or []:
                    if isinstance(i, int) and i in valid:
                        asked.append(i)
            _debug({"kind": "retrieve", "asked": asked, "left": retrievals_left})
            if asked and retrievals_left > 0:
                self._retrieved_ids.update(asked)
                retrievals_left -= 1
                continue
            # budget exhausted or all ids invalid: force an actual action
            tools = [t for t in tools if t is not _RETRIEVE_TOOL]

        # 6. record into the segment; base generate_next_message appends to state.messages
        self._segments[-1]["messages"].append(assistant_message)
        return assistant_message


def create_llm_hiagent(tools: List[Tool], domain_policy: str, **kwargs) -> LLMHiAgent:
    # kwargs may carry extra runner params (e.g. task); only forward what
    # LLMAgent.__init__ accepts, mirroring create_llm_agent.
    return LLMHiAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
