"""Offline regressions for scheduler parity, permission guards and actual reply ingestion."""

import itertools
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tau2.agent import schema_user_agent as rt
from tau2.agent.scheduler_ablation import (
    SchedulerAblationAgent,
    create_scheduler_ablation_agent,
)
from tau2.agent.schema_agent import _apply_patch_ops, eval_pred

ROOT = Path(__file__).resolve().parents[1]


class Fixture(SchedulerAblationAgent):
    def __init__(self):
        self.task_spec = {}
        self.task = SimpleNamespace(id="audit-fixture")
        self._seed = 626729
        self.subtask_spec = {
            "FIX": {"base_actions": [{"tool": "toggle_data", "when": "always"}]}
        }
        self.sequence = [
            {"name": "repair", "subtask_type": "FIX", "when": "fault == true"},
            {"name": "close", "subtask_type": "CLOSE", "when": "always"},
            {"name": "diag", "subtask_type": "DIAG", "when": "always"},
            {"name": "off", "subtask_type": "OFF", "when": "always"},
        ]
        self.idx = {
            "order": {s["name"]: i for i, s in enumerate(self.sequence)},
            "produces": {
                "FIX": {"fault"},
                "CLOSE": {"goal", "resolved"},
                "DIAG": {"fault"},
                "OFF": {"noise"},
            },
            "is_repair": {"FIX": True, "CLOSE": False, "DIAG": False, "OFF": False},
            "goal_fields": {"goal", "resolved"},
        }

    def _family_spec(self, st):
        return {
            "base_subtask_sequence": self.sequence,
            "success_when_all": ["goal == true"],
        }

    def _causal_index(self, st):
        return self.idx

    def _needed_fields(self, st, idx):
        return {"goal", "resolved", "fault"}

    def _any_repair_done(self, st, idx):
        return st.repair_attempted

    def _on_enter(self, sub, st):
        pass

    def _log_replan(self, *args):
        pass

    def _subtask_by_name(self, st, name):
        return next((s for s in self.sequence if s["name"] == name), None)


class Oracle(Fixture):
    _priority = rt.SchemaUserAgent._priority


def state(**fields):
    return rt.SchemaUserAgentState(
        system_messages=[],
        messages=[],
        family="mms",
        task_state={
            "resolved": False,
            "fault": None,
            "goal": False,
            "phone_number": "555-123-2002",
            **fields,
        },
        repair_attempted=True,
        subtask_done={},
    )


def choose(a, st, policy):
    with patch.dict(os.environ, {"SCHEMAFLEX_SCHEDULER_POLICY": policy}):
        return a._active_subtask(st)


class EndIngest(Exception):
    pass


class IngestFixture(Fixture):
    def __init__(self, ops, complete=False):
        super().__init__()
        self.ops = ops
        self.complete = complete

    def _update_from_user_reply(self, *args):
        return _apply_patch_ops(
            args[-1].task_state,
            self.ops,
            [
                {"op": "set", "path": "task_state.resolved"},
                {"op": "set", "path": "task_state.fault"},
            ],
        )

    def _mark_done_if_complete(self, pend, st):
        if self.complete:
            st.subtask_done[pend["subtask"]] = True

    def _dwa_fields(self, *args):
        return []

    def _active_subtask(self, st):
        raise EndIngest


def ingest(st, ops, complete=False):
    st.pending = {"subtask": "repair", "subtask_type": "FIX", "tool_name": "fake"}
    st.pending_kind = "user"
    with unittest.TestCase().assertRaises(EndIngest):
        IngestFixture(ops, complete).generate_next_message(
            rt.UserMessage(role="user", content="Please clarify."), st
        )


class Tests(unittest.TestCase):
    def setUp(self):
        self.flags = patch.multiple(
            rt,
            _REPLAN_ON=True,
            _THRU_ON=True,
            _THRU_V1=True,
            _DIAG_ON=False,
            _WATCHDOG_ON=False,
            _SAGE_ON=False,
        )
        self.flags.start()
        self.addCleanup(self.flags.stop)

    def test_full_uses_original_selection_method(self):
        self.assertIs(
            SchedulerAblationAgent._active_subtask_replan,
            rt.SchemaUserAgent._active_subtask_replan,
        )
        self.assertIs(
            SchedulerAblationAgent._active_subtask, rt.SchemaUserAgent._active_subtask
        )

    def test_full_matches_original_over_state_matrix(self):
        cases = 0
        for fault, goal, repaired, diagdone in itertools.product(
            [None, False, True], [None, False, True], [False, True], [False, True]
        ):
            st = state(fault=fault, goal=goal)
            st.repair_attempted = repaired
            if diagdone:
                st.subtask_done["diag"] = True
            expected = Oracle()._active_subtask_replan(st.model_copy(deep=True))
            actual = choose(Fixture(), st.model_copy(deep=True), "full")
            self.assertEqual(actual, expected)
            cases += 1
        self.assertEqual(cases, 36)

    def test_full_keeps_eligible_closer(self):
        st = state()
        st.subtask_done["repair"] = True
        revised = choose(Fixture(), st, "full")
        self.assertEqual(revised["name"], "close")

    def test_base_is_first_guard_eligible(self):
        for fault in [None, False, True]:
            a = Fixture()
            st = state(fault=fault)
            expected = next(
                s for s in a.sequence if eval_pred(s["when"], st.task_state)
            )
            self.assertEqual(choose(a, st, "base"), expected)

    def test_base_preserves_first_position_with_duplicate_names(self):
        a = Fixture()
        a.sequence.append(dict(a.sequence[1]))
        a.idx["order"] = {s["name"]: i for i, s in enumerate(a.sequence)}
        self.assertEqual(choose(a, state(), "base")["name"], "close")

    def test_fixed_keeps_declared_tiers(self):
        self.assertEqual(
            choose(Fixture(), state(fault=True), "fixed")["name"], "repair"
        )
        self.assertEqual(
            choose(Fixture(), state(fault=False), "fixed")["name"], "close"
        )

    def test_candidate_membership_unchanged_for_all_policies(self):
        for policy in ["base", "fixed", "full"]:
            a = Fixture()
            st = state(fault=False)
            st.subtask_done = {"close": True, "diag": True}
            self.assertEqual(choose(a, st, policy)["name"], "off")

    def test_sticky_execution_is_shared(self):
        for policy in ["base", "fixed", "full"]:
            st = state()
            st.active_subtask = "off"
            self.assertEqual(choose(Fixture(), st, policy)["name"], "off")

    def test_empty_candidate_set(self):
        for policy in ["base", "fixed", "full"]:
            st = state()
            st.subtask_done = {s["name"]: True for s in Fixture().sequence}
            self.assertIsNone(choose(Fixture(), st, policy))

    def test_invalid_policy_and_disabled_selector_rejected(self):
        with self.assertRaises(ValueError):
            choose(Fixture(), state(), "typo")
        with patch.object(rt, "_REPLAN_ON", False), self.assertRaises(RuntimeError):
            create_scheduler_ablation_agent([], "")

    def test_permission_entry_and_action_profiles(self):
        task = json.loads((ROOT / "memory" / "task_spec.json").read_text())
        subtasks = json.loads((ROOT / "memory" / "subtask_spec.json").read_text())
        seen = 0
        for family in task.values():
            if not isinstance(family, dict):
                continue
            for s in family.get("base_subtask_sequence", []):
                if s["name"] not in ["fixStoragePerm", "fixSmsPerm"]:
                    continue
                field = (
                    "mms_storage_perm"
                    if s["name"] == "fixStoragePerm"
                    else "mms_sms_perm"
                )
                ts = {
                    "resolved": False,
                    "service_status": "connected",
                    "speed_test": "poor",
                }
                self.assertEqual(eval_pred(s["when"], ts), False)
                action = subtasks[s["subtask_type"]]["base_actions"][0]
                self.assertEqual(eval_pred(action["when"], ts), False)
                ts[field] = False
                self.assertTrue(eval_pred(s["when"], ts))
                self.assertTrue(eval_pred(action["when"], ts))
                ts[field] = True
                self.assertFalse(eval_pred(s["when"], ts))
                self.assertFalse(eval_pred(action["when"], ts))
                seen += 1
        self.assertEqual(seen, 2)

    def test_repeated_identical_writes_regression(self):
        st = state()
        for n in range(1, 4):
            ingest(st, [{"op": "set", "path": "task_state.resolved", "value": "false"}])
            self.assertEqual(st.stuck["repair"], n)
        self.assertEqual(
            Fixture()._should_give_up(
                {"name": "repair", "subtask_type": "FIX"}, st, "test"
            ),
            True,
        )

    def test_genuine_progress_resets_counter(self):
        st = state(fault=True)
        st.stuck["repair"] = 2
        ingest(st, [{"op": "set", "path": "task_state.fault", "value": False}])
        self.assertEqual(st.stuck["repair"], 0)

    def test_completed_subtask_clears_counter(self):
        st = state()
        st.stuck["repair"] = 2
        ingest(st, [], complete=True)
        self.assertNotIn("repair", st.stuck)
