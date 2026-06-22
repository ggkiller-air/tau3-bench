# 实验报告 — Codex 执行记录

> Codex 跑完 `EXPERIMENTS.md` 里的每轮请求后，在此**新开一节**写报告（格式见 EXPERIMENTS.md 各轮的"报告格式"）。Claude 读这里决定下一步。

<!-- Codex: 从下方开始追加，例如 "## Round 1 报告" -->

## Round 1 报告

### 三臂 pass^k

| 臂 | 栈 | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---|---:|---:|---:|---:|
| A | REPLAN+CLOSE+GROUND+SHOTGUN | 0.9000 | 0.8583 | 0.8250 | 0.8000 |
| B | REPLAN+CLOSE+GROUND | 0.9250 | 0.9000 | 0.8750 | 0.8500 |
| C | A + ELICIT | 0.8500 | 0.8000 | 0.7750 | 0.7500 |

### 逐家族 pass^1

| 臂 | service | mobile_data | mms |
|---|---:|---:|---:|
| A | 0.7917 | 1.0000 | 0.8750 |
| B | 0.8333 | 1.0000 | 0.9167 |
| C | 0.8333 | 0.9062 | 0.7917 |

### 残差任务 4-trial 通过数

| 任务 | A | B | C |
|---|---:|---:|---:|
| `break_apn_mms_setting` | 4/4 | 4/4 | 4/4 |
| `overdue_bill_suspension` | 0/4 | 0/4 | 0/4 |

### `analyze_arm` 原文

`python analyze_arm.py data/simulations/user_replan_ground_shotgun_full data/simulations/user_replan_ground_full`

```text
{
  "dir": "data/simulations/user_replan_ground_shotgun_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.9,
    "pass^2": 0.8583,
    "pass^3": 0.825,
    "pass^4": 0.8
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.875,
      "ok/trials": "21/24"
    },
    "mobile_data_issue": {
      "pass^1": 1.0,
      "ok/trials": "32/32"
    },
    "service_issue": {
      "pass^1": 0.7917,
      "ok/trials": "19/24"
    }
  },
  "termination": {
    "user_stop": 78,
    "max_steps": 2
  },
  "fail_buckets": {
    "B_user_stop": 6,
    "A1_repair_no_close": 2
  },
  "fail_buckets_by_family": {
    "service_issue": {
      "B_user_stop": 5
    },
    "mms_issue": {
      "A1_repair_no_close": 2,
      "B_user_stop": 1
    }
  },
  "state_events": {
    "replan": 449,
    "exec": 789,
    "update": 766,
    "close": 31,
    "stop": 77,
    "shotgun": 167
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 88272,
    "executor": 425115,
    "updater": 517404,
    "prompt": 951311,
    "completion": 79480,
    "calls": 1717,
    "peak_prompt": 65161,
    "peak_prompt_exec": 45409,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}

=== DELTA vs data/simulations/user_replan_ground_full ===
  pass^1: 0.9250 -> 0.9000  (-0.0250)
  pass^2: 0.9000 -> 0.8583  (-0.0417)
  pass^3: 0.8750 -> 0.8250  (-0.0500)
  pass^4: 0.8500 -> 0.8000  (-0.0500)
  termination: {'user_stop': 80} -> {'user_stop': 78, 'max_steps': 2}
  fail_buckets: {'B_user_stop': 6} -> {'B_user_stop': 6, 'A1_repair_no_close': 2}
```

`python analyze_arm.py data/simulations/user_replan_ground_shotgun_elicit_full data/simulations/user_replan_ground_shotgun_full`

```text
{
  "dir": "data/simulations/user_replan_ground_shotgun_elicit_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.85,
    "pass^2": 0.8,
    "pass^3": 0.775,
    "pass^4": 0.75
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.7917,
      "ok/trials": "19/24"
    },
    "mobile_data_issue": {
      "pass^1": 0.9062,
      "ok/trials": "29/32"
    },
    "service_issue": {
      "pass^1": 0.8333,
      "ok/trials": "20/24"
    }
  },
  "termination": {
    "user_stop": 76,
    "max_steps": 4
  },
  "fail_buckets": {
    "B_user_stop": 8,
    "A2_never_repaired": 3,
    "A1_repair_no_close": 1
  },
  "fail_buckets_by_family": {
    "service_issue": {
      "B_user_stop": 4
    },
    "mobile_data_issue": {
      "A2_never_repaired": 3
    },
    "mms_issue": {
      "B_user_stop": 4,
      "A1_repair_no_close": 1
    }
  },
  "state_events": {
    "replan": 459,
    "exec": 866,
    "update": 838,
    "stop": 80,
    "close": 24,
    "shotgun": 176,
    "elicit": 4,
    "elicit_reply": 4
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 88117,
    "executor": 467995,
    "updater": 555171,
    "prompt": 1028254,
    "completion": 83029,
    "calls": 1862,
    "peak_prompt": 65153,
    "peak_prompt_exec": 45542,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}

=== DELTA vs data/simulations/user_replan_ground_shotgun_full ===
  pass^1: 0.9000 -> 0.8500  (-0.0500)
  pass^2: 0.8583 -> 0.8000  (-0.0583)
  pass^3: 0.8250 -> 0.7750  (-0.0500)
  pass^4: 0.8000 -> 0.7500  (-0.0500)
  termination: {'user_stop': 78, 'max_steps': 2} -> {'user_stop': 76, 'max_steps': 4}
  fail_buckets: {'B_user_stop': 6, 'A1_repair_no_close': 2} -> {'B_user_stop': 8, 'A2_never_repaired': 3, 'A1_repair_no_close': 1}
```

`python analyze_arm.py data/simulations/user_replan_ground_full`

```text
{
  "dir": "data/simulations/user_replan_ground_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.925,
    "pass^2": 0.9,
    "pass^3": 0.875,
    "pass^4": 0.85
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.9167,
      "ok/trials": "22/24"
    },
    "mobile_data_issue": {
      "pass^1": 1.0,
      "ok/trials": "32/32"
    },
    "service_issue": {
      "pass^1": 0.8333,
      "ok/trials": "20/24"
    }
  },
  "termination": {
    "user_stop": 80
  },
  "fail_buckets": {
    "B_user_stop": 6
  },
  "fail_buckets_by_family": {
    "service_issue": {
      "B_user_stop": 4
    },
    "mms_issue": {
      "B_user_stop": 2
    }
  },
  "state_events": {
    "replan": 583,
    "exec": 803,
    "update": 778,
    "close": 29,
    "stop": 70,
    "replan_skip": 4,
    "stuck_skip": 4
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 89039,
    "executor": 428366,
    "updater": 519737,
    "prompt": 955984,
    "completion": 81158,
    "calls": 1737,
    "peak_prompt": 65211,
    "peak_prompt_exec": 44662,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}
```

### `analyze_tokens` 原文

`python analyze_tokens.py data/simulations/user_replan_ground_shotgun_full`

```text
=== 全体 80 sims: pass 72 / fail 8 ===
  PASS: calls均=15.4  peak_exec均=562  peak_any均=618  total均=8375
  FAIL: calls均=30.2  peak_exec均=615  peak_any均=660  total均=17312

=== pass^k 衰减源: 3 个 task 在 trial 间 pass/fail 混合 ===
                                       pass 3/4 | PASS peak_exec=557 calls=16.0  vs  FAIL peak_exec=597 calls=59.0  FAIL_term=['user_stop']
                                       pass 2/4 | PASS peak_exec=557 calls=34.5  vs  FAIL peak_exec=557 calls=62.0  FAIL_term=['max_steps', 'max_steps']
                                       pass 3/4 | PASS peak_exec=574 calls=13.7  vs  FAIL peak_exec=531 calls=11.0  FAIL_term=['user_stop']
```

### 结论

- 100 步本身已经把 `break_apn_mms_setting` 从残差变成稳定 `4/4`，三臂全过；真正残差只剩 `overdue_bill_suspension`，三臂都是 `0/4`，ELICIT 没有把它救起来。
- SHOTGUN 在 100 步下是负增量：A 相比 B，`pass^1 -0.0250`、`pass^4 -0.0500`，还额外引入了 2 个 `max_steps` / `A1_repair_no_close`。100 步有余量后，B 才是当前最佳臂。
- ELICIT 明显应去掉：C 相比 A 再掉 `pass^1 -0.0500`、`pass^4 -0.0500`，`elicit` 只触发了 4 次，但没救 `overdue_bill_suspension`，反而把 mobile_data 打坏，出现 `data_mode_off 1/4`、`A2_never_repaired=3`、总 `max_steps=4`。
- A 的 pass^4 仍比 pass^1 低 `0.10`，衰减依旧明显。`analyze_tokens` 显示失败 trial 的 `calls` 和 `total tokens` 显著高于成功 trial（`30.2 vs 15.4`、`17312 vs 8375`），`peak_prompt_exec` 也更高（`615 vs 562`），说明衰减仍和长对话/高调用数强相关。
- 建议下一步以 B (`REPLAN+CLOSE+GROUND`) 为基线，定点修 `overdue_bill_suspension` 的支付授权链路；不要再保留 SHOTGUN/ELICIT 进入下一轮。

### 异常记录

- 三臂 `infrastructure_error` 计数均为 `0`。
- 运行前发现父 shell 和 `.env` 带了 `http_proxy/https_proxy/all_proxy`；若直接继承会让 LiteLLM 在导入时因缺 `socksio` 崩溃。我用 `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u no_proxy -u NO_PROXY bash run_*.sh` 跑通三臂，未修改仓库脚本。

## Round 2 报告

### 三臂 pass^k

| Arm | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---:|---:|---:|---:|
| B | 0.9250 | 0.9000 | 0.8750 | 0.8500 |
| B+E | 0.8875 | 0.8500 | 0.8250 | 0.8000 |
| B+E+S | 0.8500 | 0.7667 | 0.7250 | 0.7000 |

### 逐家族 pass^1

| Arm | service_issue | mobile_data_issue | mms_issue |
|---|---:|---:|---:|
| B | 0.8333 (20/24) | 1.0000 (32/32) | 0.9167 (22/24) |
| B+E | 0.8333 (20/24) | 1.0000 (32/32) | 0.7917 (19/24) |
| B+E+S | 1.0000 (24/24) | 1.0000 (32/32) | 0.5000 (12/24) |

### `overdue_bill_suspension` 4-trial 通过数

| Arm | pass |
|---|---:|
| B | 0/4 |
| B+E | 0/4 |
| B+E+S | 4/4 |

### `analyze_arm` DELTA 原文

`python analyze_arm.py data/simulations/user_replan_ground_elicit_full data/simulations/user_replan_ground_full`

```text
{
  "dir": "data/simulations/user_replan_ground_elicit_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.8875,
    "pass^2": 0.85,
    "pass^3": 0.825,
    "pass^4": 0.8
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.7917,
      "ok/trials": "19/24"
    },
    "mobile_data_issue": {
      "pass^1": 1.0,
      "ok/trials": "32/32"
    },
    "service_issue": {
      "pass^1": 0.8333,
      "ok/trials": "20/24"
    }
  },
  "termination": {
    "user_stop": 80
  },
  "fail_buckets": {
    "B_user_stop": 9
  },
  "fail_buckets_by_family": {
    "service_issue": {
      "B_user_stop": 4
    },
    "mms_issue": {
      "B_user_stop": 5
    }
  },
  "state_events": {
    "replan": 594,
    "exec": 807,
    "update": 787,
    "close": 32,
    "stop": 94,
    "elicit": 4,
    "elicit_reply": 4,
    "replan_skip": 3,
    "stuck_skip": 3
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 89604,
    "executor": 433886,
    "updater": 530865,
    "prompt": 971929,
    "completion": 82426,
    "calls": 1756,
    "peak_prompt": 65876,
    "peak_prompt_exec": 44923,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}

=== DELTA vs data/simulations/user_replan_ground_full ===
  pass^1: 0.9250 -> 0.8875  (-0.0375)
  pass^2: 0.9000 -> 0.8500  (-0.0500)
  pass^3: 0.8750 -> 0.8250  (-0.0500)
  pass^4: 0.8500 -> 0.8000  (-0.0500)
  termination: {'user_stop': 80} -> {'user_stop': 80}
  fail_buckets: {'B_user_stop': 6} -> {'B_user_stop': 9}
```

`python analyze_arm.py data/simulations/user_replan_ground_elicit_seq_full data/simulations/user_replan_ground_elicit_full`

```text
{
  "dir": "data/simulations/user_replan_ground_elicit_seq_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.85,
    "pass^2": 0.7667,
    "pass^3": 0.725,
    "pass^4": 0.7
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.5,
      "ok/trials": "12/24"
    },
    "mobile_data_issue": {
      "pass^1": 1.0,
      "ok/trials": "32/32"
    },
    "service_issue": {
      "pass^1": 1.0,
      "ok/trials": "24/24"
    }
  },
  "termination": {
    "user_stop": 73,
    "max_steps": 7
  },
  "fail_buckets": {
    "B_user_stop": 5,
    "A1_repair_no_close": 3,
    "A2_never_repaired": 4
  },
  "fail_buckets_by_family": {
    "mms_issue": {
      "B_user_stop": 5,
      "A1_repair_no_close": 3,
      "A2_never_repaired": 4
    }
  },
  "state_events": {
    "replan": 531,
    "exec": 982,
    "update": 952,
    "close": 36,
    "stop": 50,
    "elicit": 4,
    "elicit_reply": 4,
    "replan_skip": 1,
    "stuck_skip": 1
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 88943,
    "executor": 526121,
    "updater": 630363,
    "prompt": 1154804,
    "completion": 90623,
    "calls": 2101,
    "peak_prompt": 64982,
    "peak_prompt_exec": 44298,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}

=== DELTA vs data/simulations/user_replan_ground_elicit_full ===
  pass^1: 0.8875 -> 0.8500  (-0.0375)
  pass^2: 0.8500 -> 0.7667  (-0.0833)
  pass^3: 0.8250 -> 0.7250  (-0.1000)
  pass^4: 0.8000 -> 0.7000  (-0.1000)
  termination: {'user_stop': 80} -> {'user_stop': 73, 'max_steps': 7}
  fail_buckets: {'B_user_stop': 9} -> {'B_user_stop': 5, 'A1_repair_no_close': 3, 'A2_never_repaired': 4}
```

`python analyze_arm.py data/simulations/user_replan_ground_elicit_seq_full data/simulations/user_replan_ground_full`

```text
{
  "dir": "data/simulations/user_replan_ground_elicit_seq_full",
  "n_sims": 80,
  "passk": {
    "pass^1": 0.85,
    "pass^2": 0.7667,
    "pass^3": 0.725,
    "pass^4": 0.7
  },
  "per_family_pass1": {
    "mms_issue": {
      "pass^1": 0.5,
      "ok/trials": "12/24"
    },
    "mobile_data_issue": {
      "pass^1": 1.0,
      "ok/trials": "32/32"
    },
    "service_issue": {
      "pass^1": 1.0,
      "ok/trials": "24/24"
    }
  },
  "termination": {
    "user_stop": 73,
    "max_steps": 7
  },
  "fail_buckets": {
    "B_user_stop": 5,
    "A1_repair_no_close": 3,
    "A2_never_repaired": 4
  },
  "fail_buckets_by_family": {
    "mms_issue": {
      "B_user_stop": 5,
      "A1_repair_no_close": 3,
      "A2_never_repaired": 4
    }
  },
  "state_events": {
    "replan": 531,
    "exec": 982,
    "update": 952,
    "close": 36,
    "stop": 50,
    "elicit": 4,
    "elicit_reply": 4,
    "replan_skip": 1,
    "stuck_skip": 1
  },
  "supervisor": {
    "total_calls": 0,
    "decisions": {},
    "schedule_valid_target": "0/0",
    "extract_fields_written": 0,
    "reschedule_events": 0,
    "escalate_events": 0
  },
  "tokens": {
    "macro": 88943,
    "executor": 526121,
    "updater": 630363,
    "prompt": 1154804,
    "completion": 90623,
    "calls": 2101,
    "peak_prompt": 64982,
    "peak_prompt_exec": 44298,
    "seed": 27270060
  },
  "tokens_note": "supervisor token 桶若缺失=_ROLE_BY_CALL 未含 schema_supervisor（当前不记账）；supervisor 调用次数以 state_events.supervise 为准"
}

=== DELTA vs data/simulations/user_replan_ground_full ===
  pass^1: 0.9250 -> 0.8500  (-0.0750)
  pass^2: 0.9000 -> 0.7667  (-0.1333)
  pass^3: 0.8750 -> 0.7250  (-0.1500)
  pass^4: 0.8500 -> 0.7000  (-0.1500)
  termination: {'user_stop': 80} -> {'user_stop': 73, 'max_steps': 7}
  fail_buckets: {'B_user_stop': 6} -> {'B_user_stop': 5, 'A1_repair_no_close': 3, 'A2_never_repaired': 4}
```

### elicit 触发计数

- `user_replan_ground_elicit_full`: `4`
- `user_replan_ground_elicit_seq_full`: `4`

### B+E+S vs B+E 逐任务 diff

- `[service_issue]overdue_bill_suspension[PERSONA:Easy]`: `0/4 -> 4/4`
- `[mms_issue]bad_network_preference[PERSONA:None]`: `4/4 -> 1/4`
- `[mms_issue]bad_wifi_calling[PERSONA:Easy]`: `4/4 -> 2/4`
- `[mms_issue]break_apn_mms_setting[PERSONA:Hard]`: `4/4 -> 2/4`

### 结论

- `overdue_bill_suspension` 已被 `L-SEQ` 修掉：`B/B+E/B+E+S = 0/4 / 0/4 / 4/4`，说明 ELICIT 只解第一层，SEQ 确实是必要的第二层。
- 但 `L-SEQ` 没有满足“零回归”：`B+E+S` 相对 `B+E` 把 3 个 MMS 任务打坏，尤其 `bad_network_preference 4/4 -> 1/4`，导致整体 `pass^1/pass^4` 反而再掉 `-0.0375/-0.1000`。
- ELICIT 在 B 上单独看仍是负增量：`B+E vs B` 的 `pass^1 -0.0375`、`pass^4 -0.0500`；两臂 `elicit=4`，触发次数与 overdue 的 4 个 trial 一致，没有污染到 mobile_data/mms 的触发计数。
- Round 2 结束后新的最佳栈仍是 **B = REPLAN+CLOSE+GROUND**，不是 `B+E` 也不是 `B+E+S`。如果要保留 overdue 修复收益，下一步应定点排查 SEQ 对 MMS 多步修复链路的副作用，而不是直接把当前 SEQ 合入主线。

### 异常记录

- `run_replan_ground_elicit_full.sh` 和 `run_replan_ground_elicit_seq_full.sh` 以 `nohup bash ... > logs/*.nohup.log 2>&1 &` 启动时，壳进程都会很快退出，`nohup` 日志只留下 start 行，不会持续承载 `tau2 run` 输出；为拿到完整结果，我改为串行直接执行脚本并等待其最终 `done rc=0`，全程仍保证只有一个 `tau2 run` 在占用 GPU0。
- 两个新臂都完成且 `done rc=0`；未见 `infrastructure_error` 检查需求中的显式报错，但 `B+E+S` 出现了 `max_steps=7`，全部集中在 `mms_issue`。

## Round 2b 报告（L-SEQ 修正版，Claude 直跑核验）

> Round 2 的 SEQ v1（不分轴）虽 overdue 0/4→4/4，但 R-skip 把会 balk 的 user 侧写动作也跳过 → mms 死循环（can_send_mms 被调 24/55 次）→ mms 19→12、net 负。迭代两版定位到正确区分轴。

**三版 L-SEQ 迭代**（B+E+S 臂，对照 B+E / 纯 B）：
| 版本 | 区分轴 | pass^1 | pass^4 | overdue | mms 家族 | max_steps |
|---|---|---|---|---|---|---|
| v1 | 不分轴（all base_actions） | 0.850 | 0.700 | 4/4 | 12/24 | 7 |
| v2 | agent vs user（错轴：payment 其实是 user 侧） | 0.875 | 0.750 | 0/4 | 20/24 | 1 |
| **v3** | **只读 vs 写（`_READONLY_TOOL_RE`/`_MUT_TOOL_RE`）** | **0.975** | **0.900** | **4/4** | **23/24** | **0** |

**v3 = 新最佳栈 B+ELICIT+SEQ**（0 infra_error，elicit 触发 5）：
```
=== DELTA vs data/simulations/user_replan_ground_full (纯 B) ===
  pass^1: 0.9250 -> 0.9750  (+0.0500)
  pass^2: 0.9000 -> 0.9500  (+0.0500)
  pass^3: 0.8750 -> 0.9250  (+0.0500)
  pass^4: 0.8500 -> 0.9000  (+0.0500)
  fail_buckets: {'B_user_stop': 6} -> {'B_user_stop': 2}
```
逐家族 pass^1：service 24/24、mms 23/24、mobile_data 31/32。逐任务 vs B+E：overdue 0→4、break_app_both 1→4、break_app_sms 3→4、break_app_storage 3→4；bad_wifi 4→3、data_mode 4→3（±1 trial 噪声）。

**关键洞见（写 paper 用）**：L-SEQ 的正确区分轴是**只读门 vs mutating 写**，不是 agent/user（check_payment_request、make_payment 都是 user 侧）。只读门（when 永不翻）需 at-most-once 跳过以推进到终末写；mutating 写会被用户 balk、必须保留重发。R-doneguard 只在终末是 mutating 写时生效（治 makePayment 提前 done），对 FIX_* 的只读 retest 终末自动关闭。

**保留**：单次 4-trial，±0.05 噪声；overdue +4 是确定性结构修复（铁），mms/mobile 小波动是噪声，建议补一次确认跑坐实 0.975。
