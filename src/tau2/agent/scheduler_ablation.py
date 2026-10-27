"""Ranking-only ablation over the runtime's unchanged admissible set.

Full delegates to the original priority function. The runtime keeps its original
sticky execution and all shared state/reply/provenance controls. No new candidate
filter is installed by this experiment wrapper.
"""

import os

from tau2.agent import schema_user_agent as runtime


class SchedulerAblationAgent(runtime.SchemaUserAgent):
    """Select Base, Fixed or Full ranking while sharing the original executor."""

    def _priority(self, state, idx, needed, subtask):
        policy = os.environ["SCHEMAFLEX_SCHEDULER_POLICY"]
        if policy == "full":
            return super()._priority(state, idx, needed, subtask)
        if policy == "base":
            # Use the first actual schema position, even if an identical named
            # subtask appears twice. This implements first guard-eligible exactly.
            sequence = self._family_spec(state)["base_subtask_sequence"]
            return (0, -sequence.index(subtask))
        if policy == "fixed":
            kind = subtask["subtask_type"]
            tier = (
                4
                if idx["is_repair"][kind]
                else (3 if "resolved" in idx["produces"][kind] else 2)
            )
            return (tier, -idx["order"][subtask["name"]])
        raise ValueError(f"Unknown scheduler policy: {policy}")


def create_scheduler_ablation_agent(tools, domain_policy, **kwargs):
    """Build the registered scheduler ablation with replanning enabled."""
    if not runtime._REPLAN_ON:
        raise RuntimeError("Ranking ablation requires SCHEMAFLEX_REPLAN=1 in every arm")
    return SchedulerAblationAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
