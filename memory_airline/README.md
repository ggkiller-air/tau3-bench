# memory_airline — SchemaFlex airline dialogue schema

Hand-authored `task_spec.json` + `subtask_spec.json` for the tau2 **airline** domain,
run by the **unified `schema_user_agent`** (the same agent that runs telecom). Authored from
`data/tau2/domains/airline/{policy.md, tools.py, data_model.py}` against the dialogue
runtime contract in `.claude/skills/telecom-schema/reference/`.

## One agent, domain text in the schema

Airline is **dialogue-only** (0 tickets; all 50 tasks are `user_scenario`) and structurally
different from telecom (it has no user-side device tools — the agent holds *every* tool; the
customer only supplies *information* and *confirmation*). Rather than fork a new agent, the three
telecom-specific assumptions in `schema_user_agent` were made **schema-driven, with the telecom
constants as defaults** (so telecom stays byte-identical). The airline schema supplies its own
values via these family-level keys in `task_spec.json`:

1. **`identity_field` / `identity_label` / `identity_prompt`** — telecom bootstraps on
   `phone_number`; airline sets `user_id`. The base agent reads the field/prompt from the schema.
2. **`agent_prompts.instruct_system`** — the device-action instruction prompt is replaced by an
   airline ask-for-info / confirm-before-write prompt.
3. **`agent_prompts.executor_system`** — an airline tool-arg-assembly prompt.

Everything else — the guarded/sticky walk, `_classify` agent/user routing, the dialogue
StateUpdater, give-up + WATCHDOG/REPLAN/ELICIT levers — is inherited unchanged. Telecom omits
these keys and keeps its defaults: **one agent serves both domains.**

## Design

**Single family `airline`** (task ids are bare integers → `_pick_family` falls back to the only
family). **Intent** (`cancel`/`compensation`/`modify_baggage`/`book`/`modify_flights`/…) is
seeded from the opening narrative (`init_rules`) or asked by `ASK_INTENT`, then gates the subtree.

```
IDENTIFY_USER → ASK_INTENT? → GET_RESERVATION (captures res_flights for reassembly)
  ├ cancel:        ASK_CANCEL_REASON → CONFIRM_CANCEL → DO_CANCEL   | REFUSE_REQUEST  | (flown) TRANSFER
  ├ compensation:  CHECK_COMPENSATION → OFFER_COMPENSATION → DO_COMPENSATION | REFUSE_REQUEST
  ├ modify_baggage:ASK_BAGGAGE → CONFIRM_BAGGAGE → DO_BAGGAGE
  ├ modify_cabin:  ASK_CABIN_CHANGE → CONFIRM_MODIFY → DO_MODIFY_FLIGHTS   (flights kept verbatim, cabin swapped)
  ├ modify_flights:ASK_FLIGHT_CHANGE → CONFIRM_MODIFY → DO_MODIFY_FLIGHTS  | (basic_economy) REFUSE_REQUEST
  └ book / modify_passengers / other → TRANSFER_HUMAN
```

The cabin/flight change reuses the reservation's current `res_flights` (the full segment array
`update_reservation_flights` requires) — verbatim for a cabin-only change, with the changed
segment swapped for a flight change.

Every write is gated behind an explicit **confirmation** subtask (policy: list details + get a
yes before any DB write). `resolved` is set only by terminal subtasks (a write landed, a refusal
was stated, or the customer declined); `escalated` is set by `transfer_to_human_agents`.

### Policy baked into `parse_rules` (the schema owns the procedure)

- **Cancel eligibility** (`ASK_CANCEL_REASON` → `cancel_allowed`): allowed iff booked ≤24h ago
  **or** `airline_cancelled` **or** `business` cabin **or** (`insurance==yes` and reason is
  `health`/`weather`). Any already-flown segment → `TRANSFER_HUMAN` (gated upstream).
- **Compensation eligibility** (`CHECK_COMPENSATION` → `comp_eligible`): only if `silver`/`gold`
  **or** `insurance==yes` **or** `business`. Amount = `100×passengers` (cancelled) / `50×passengers`
  (delayed). Never proactively offered (the customer must ask).
- **Baggage free allowance** (`ASK_BAGGAGE` → `bag_nonfree`): per-passenger table by
  membership×cabin; paid bags = `max(0, total − free)` at $50 each.
- **Dates** are computed against the fixed in-domain current time **2024-05-15 15:00 EST**.

## Coverage (v1) and honest expectations

**Implemented:** identify → reservation → **cancel / compensation / modify_baggage / modify_cabin
/ modify_flights / refuse / transfer-when-flown**.

- `modify_cabin` is the most tractable write (reuse `res_flights` verbatim, swap cabin).
- `modify_flights` is built but hard: tasks that need the agent to *search and pick* a flight
  ("cheapest", "nonstop") or to downgrade *multiple* reservations remain low-yield — the 8B must
  do the search/selection/array-assembly itself; the schema can only structure the steps.

**Still routed to `TRANSFER_HUMAN`:** `book` (largest structured payload), `modify_passengers`,
`other`. `book_reservation` is the next build.

**Score reality:** eval is strict (every task scored on **DB and COMMUNICATE**). Telecom's
*dialogue* ceiling was ~45% (8B) / ~55% (gpt-5.4 user-sim), with the residual attributed to
dynamic control. Airline is harder, so v1 is a correctness-first baseline to iterate from — not
a high score out of the gate. The book/modify-flights tasks are an explicit known gap.

## Run

```bash
# 1) local 8B must be up on :9000 (2-GPU server — do NOT touch :8000):
bash scripts/start_2gpu_server.sh --model qwen3-8b --port 9000

# 2) smoke (3 tasks), then iterate on the 20-task test split, then report all 50:
scripts/runUserAirline.sh air_smoke --task-split-name test --num-tasks 3
scripts/runUserAirline.sh air_test  --task-split-name test
scripts/runUserAirline.sh air_full
```

Agent = `schema_user_agent` (qwen3-8b @ :9000); user-sim = `gpt-5.4` via `.env` proxy.
`SCHEMAFLEX_SPEC_DIR=memory_airline` and `SCHEMAFLEX_WATCHDOG=1` are set by the runner. State log
→ `data/simulations/<run>/state_log.jsonl`; report → `<run>/report.html`.

## Iteration notes

- Triage with `state_log.jsonl`: each `exec`/`update`/`stop` record shows the subtask, the
  customer reply / tool result, the emitted `patch_ops`, and the resulting `task_state` — diff
  the reply against what landed to find a mis-extraction.
- Common levers (env, off by default): `SCHEMAFLEX_REPLAN=1` (causal re-ranker),
  `SCHEMAFLEX_ELICIT=1` (ask for a null context precondition), `SCHEMAFLEX_SEQ=1` (ordered
  multi-action guard). Add per-A/B as failures dictate; don't hand-tune endlessly.
- Next build: real `book` and `modify_flights` subtrees (search_*_flight → assemble payload →
  confirm → write), which will need careful executor arg-assembly rules and likely a HILO
  big-model corrector on the write step.
