"""Plain ReAct solo agent for standard (scenario-based) tau2 domains.

This is the *no-schema* baseline for the retail/airline generalization study. The
stock `llm_agent_solo` (LLMSoloAgent) requires `task.ticket`; scenario-based domains
(retail) carry `user_scenario` instead, so those tasks fail `check_valid_task` and get
filtered out — which is why a naive-ReAct baseline was never measurable on retail.

This thin subclass only:
  1. accepts no-ticket tasks that carry evaluation actions, and
  2. synthesizes a ticket from `user_scenario` so the inherited system prompt works.

Everything else (full domain policy in the system prompt, all backend tools, standard
ReAct loop) is unchanged. It exists purely to establish the floor (8B) and ceiling
(gpt-5.4) against which the schema agent's value is judged. It does NOT touch telecom.
"""

from __future__ import annotations

from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import MultiToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.utils.llm_utils import generate


class LLMSoloStdAgent(LLMSoloAgent):
    """Plain ReAct solo agent that accepts scenario-based (no-ticket) tasks.

    Adds a neutral kickoff user turn on the first step. The stock solo agent makes its
    first model call with a system message and NO user turn; the micuapi/Responses-API
    proxy that serves gpt-5.4 rejects that ("One of input/... must be provided"). The
    kickoff carries no task content (policy + ticket already live in the system prompt),
    so it only satisfies the wire and does not change what the baseline knows.
    """

    KICKOFF = (
        "Please handle the customer's request as described in the <ticket> in your "
        "instructions. Use the available tools to make the required changes, then call "
        "`done` when the task is fully complete."
    )

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        if getattr(task, "ticket", None) is None:
            if getattr(task, "user_scenario", None) is None:
                return False
            if task.evaluation_criteria is None:
                return False
            return bool(task.evaluation_criteria.actions or [])
        return super().check_valid_task(task)

    def __init__(self, tools, domain_policy, task, llm, llm_args=None):
        if getattr(task, "ticket", None) is None and getattr(task, "user_scenario", None) is not None:
            task.ticket = str(task.user_scenario)
        super().__init__(
            tools=tools, domain_policy=domain_policy, task=task, llm=llm, llm_args=llm_args
        )

    def get_init_state(self, message_history=None):
        state = super().get_init_state(message_history)
        if not state.messages:
            state.messages.append(UserMessage(role="user", content=self.KICKOFF))
        return state

    def generate_next_message(self, message, state):
        # Same loop as LLMSoloAgent, but tolerant of the kickoff user turn seeded in
        # get_init_state (the parent asserts an empty history when message is None and
        # raises on any UserMessage).
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None and not isinstance(message, UserMessage):
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            tool_choice="required",
            call_name="agent_solo_response",
            **self.llm_args,
        )
        if not assistant_message.is_tool_call():
            raise ValueError("LLMSoloStdAgent only supports tool calls.")
        assistant_message = self._check_if_stop_toolcall(assistant_message)
        state.messages.append(assistant_message)
        return assistant_message, state


def create_llm_solo_std_agent(tools, domain_policy, **kwargs):
    return LLMSoloStdAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
