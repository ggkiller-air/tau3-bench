"""SLM-ReAct: text-format ReAct baseline over the stock LLMAgent.

The tau2 default agent (LLMAgent, our "SLM-FC" row) uses NATIVE function calling.
This agent is the classic text-format ReAct baseline instead (tau-bench's original
"react" strategy): tools are described in the system prompt, the model emits

    Thought: <free-form reasoning>
    Action: {"name": <tool_name>, "arguments": {...}}
or
    Thought: <free-form reasoning>
    Respond: <message to the user>

as PLAIN TEXT with no tool schema passed to the API; the agent parses the text into
a proper ToolCall (or user-facing message) for the orchestrator. The LLM-side
conversation is kept fully textual: its own turns are stored as raw ReAct text and
incoming tool results are stored as "Observation:" user messages, so the model always
sees (and continues) the ReAct trace format.

Parsing is deliberately single-shot with a text fallback (an unparseable output is
sent to the user verbatim, minus the think block): no hidden retry calls, so every
LLM call of this baseline lands in results.json and token accounting needs no sidecar.
"""

import json
import re
import uuid
from typing import List, Optional

from loguru import logger

from tau2.agent.llm_agent import SYSTEM_PROMPT, LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

REACT_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You interact by producing, at every turn, EXACTLY ONE of the two blocks below and nothing else:

Thought: <your step-by-step reasoning, kept brief>
Action: {{"name": "<tool_name>", "arguments": {{<JSON arguments>}}}}

or

Thought: <your step-by-step reasoning, kept brief>
Respond: <the message to send to the user>

Rules:
- "Action" invokes one of the tools listed in <tools>. The value after "Action:" must be a single valid JSON object on one line.
- "Respond" sends a message to the user. You cannot call a tool and respond in the same turn.
- After an Action you will receive an "Observation:" with the tool result.
- Always follow the policy.

<tools>
{tool_docs}
</tools>
""".strip()

# Qwen3 emits <think>...</think> by default; it must not leak into parsing or the
# user-facing fallback message.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\{.*)", re.DOTALL)
_RESPOND_RE = re.compile(r"Respond:\s*(.*)", re.DOTALL)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "")


def _parse_first_json_object(text: str) -> Optional[dict]:
    """Parse the first complete JSON object at the start of `text` (brace matching,
    tolerant of trailing garbage such as a stray closing fence or a second block)."""
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class LLMReActAgent(LLMAgent):
    """LLMAgent variant speaking the classic text ReAct protocol instead of native FC."""

    @property
    def system_prompt(self) -> str:
        tool_docs = "\n".join(
            json.dumps(t.openai_schema, ensure_ascii=False) for t in self.tools
        )
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=REACT_AGENT_INSTRUCTION.format(tool_docs=tool_docs),
        )

    def _to_react_text(self, message) -> UserMessage:
        """Render an incoming tool result as an Observation user-turn (keeps the
        LLM-side conversation purely textual)."""
        if isinstance(message, MultiToolMessage):
            body = "\n".join(
                f"Observation: {tm.content or ''}" for tm in message.tool_messages
            )
        else:
            body = f"Observation: {message.content or ''}"
        return UserMessage(role="user", content=body)

    def _generate_next_message(
        self, message, state: LLMAgentState
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, (ToolMessage, MultiToolMessage)):
            state.messages.append(self._to_react_text(message))
        else:
            state.messages.append(message)
        raw = generate(
            model=self.llm,
            messages=state.system_messages + state.messages,
            call_name="agent_response",
            **self.llm_args,
        )
        text = _strip_think(raw.content or "")

        m = _ACTION_RE.search(text)
        if m:
            call = _parse_first_json_object(m.group(1))
            if call and isinstance(call.get("name"), str):
                args = call.get("arguments")
                tool_call = ToolCall(
                    id=f"react_{uuid.uuid4().hex[:24]}",
                    name=call["name"],
                    arguments=args if isinstance(args, dict) else {},
                )
                # LLM-side history keeps the raw ReAct text; the orchestrator gets a
                # pure tool-call message (no content — mixed messages are rejected).
                state.messages.append(
                    AssistantMessage(role="assistant", content=text)
                )
                out = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                    usage=raw.usage,
                    cost=raw.cost,
                    raw_data=raw.raw_data,
                )
                out.generation_time_seconds = raw.generation_time_seconds
                return out
            logger.warning("ReAct Action block failed to parse; falling back to Respond")

        m = _RESPOND_RE.search(text)
        if m:
            reply = m.group(1).strip()
        else:
            # Fallback: the model wrote the user-facing content inside (or after) the
            # Thought block without a Respond marker — strip the internal label only.
            reply = re.sub(r"^\s*Thought:\s*", "", text).strip()
        reply = reply or "Could you clarify?"
        state.messages.append(AssistantMessage(role="assistant", content=text))
        out = AssistantMessage(
            role="assistant",
            content=reply,
            usage=raw.usage,
            cost=raw.cost,
            raw_data=raw.raw_data,
        )
        out.generation_time_seconds = raw.generation_time_seconds
        return out

    def generate_next_message(
        self, message, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        # The base class appends the returned message to state.messages; here the
        # LLM-side history was already updated with the raw ReAct text inside
        # _generate_next_message, so the parsed message must NOT be appended again.
        assistant_message = self._generate_next_message(message, state)
        return assistant_message, state


def create_llm_react_agent(
    tools: List[Tool], domain_policy: str, **kwargs
) -> LLMReActAgent:
    # kwargs may carry extra runner params (e.g. task); only forward what
    # LLMAgent.__init__ accepts, mirroring create_llm_agent.
    return LLMReActAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
