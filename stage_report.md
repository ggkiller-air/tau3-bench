# SchemaFlex / DyLight · τ³-bench telecom user 模式 — 阶段成果报告

> 实验数据见 `stage_report.html`（自包含单页，浏览器直接打开）。本文档为技术路线、诊断与下一步。
> 2026-06-22 · 高低协同 agent：DyLight 动态调度 + SchemaFlex schema · 本地 qwen3-8b(agent) + gpt-5.4(user-sim)

---

## 一句话结论

在 τ³-bench telecom **user 模式**（该任务最难设定）上，一组**纯编排层 lever** 把 telecom_small pass^1 从 baseline 0.35 推到 **0.975**、pass^4 到 **0.90**（目标 ≥0.90/≥0.80 已达成并超额）；在极难复合集 strat90 上泛化到 **0.644**。

---

## 0 · 背景与设定

**τ³-bench telecom user 模式**是该 benchmark 最难的设定，三个约束叠加：

- **工具非对称**：agent 只动后台/计费；**只有用户能点手机**（toggle_roaming / run_speed_test / grant_app_permission / reset_apn …）。排障 = agent 多轮引导 LLM-user 操作手机 + 解读模糊回复。
- **一次只能做一个动作**：长程交互被切碎，轮数/步数是硬约束。
- **二值评测、无部分分**：reward 看最终 DB/env 状态匹配，撞 max_steps 直接判 0、根本不评断言。

**方法论原则**（贯穿全程）：

1. 所有调度逻辑活在自定义 agent 内，**主循环 / user / env / 评测一律不改**。
2. 每个 lever 挂独立环境变量开关、**OFF 字节一致**，可干净消融。
3. 增益判定靠**逐任务 + state_log 事件归因 + 控噪**，不看单跑 raw pass^k（gpt-5.4 user-sim 非确定，单次 4-trial 有 ±0.05 噪声）。

---

## 1 · 技术路线 — 一组纯编排层 lever

```
REPLAN → CLOSE → GROUND → ELICIT → SEQ
```

每个 lever 针对一个被实证定位的失败机制（数据见 HTML §1 消融表）：

| Lever | 机制（治什么） | 一句话做法 |
|---|---|---|
| **REPLAN** | 静态 schema 走查把诊断/修复顺序写死 | 按因果链(produces/consumes)动态重排，修复先行 + 剪死分支 |
| **CLOSE** | 已解决却撞 max_steps 被硬归零 | `success_when_all` 满足即主动收尾（护栏：必须真 dispatch 过 mutating 工具） |
| **GROUND** | agent↔user 指令接缝（复合句/UI 措辞用户接不住） | 单动作 + 用 tool/args 字面精确命名，滤掉多步脚手架 |
| **ELICIT** | 上下文前置字段（pay_allowed/is_abroad）schema 恒填 null | give-up 前问用户 yes/no 补上，解析填字段 |
| **SEQ** | 只读门→写 的序列修复死锁 | 跳过已跑只读门推进到终末写 + 末位写未跑不判 done |

### 关键转折一：放开步数推翻了 40 步时代的最佳栈

40 步时代 SHOTGUN（mms 盲修全试）是突破性 lever。但把 max_steps 40→100 重测后发现：**SHOTGUN 的全部价值来自「40 步装不下诊断+修复+retest」这个前提**，步数放开后盲修反成负担（额外占步数），最简栈 B=REPLAN+CLOSE+GROUND 反而最优（0.925/0.85）。残差 `break_apn_mms`（曾经的「40 步硬墙」）也随步数放开自己消失。→ SHOTGUN 已从主线摘除。这是个科研诚实的反转：**lever 的价值依赖约束条件，约束变了要重测**。

### 关键转折二：L-SEQ 的「只读 vs 写」区分轴（迭代 3 版才定位）

攻克最后残差 `overdue_bill_suspension` 时，L-SEQ 经历 3 版迭代，暴露出一个有普适性的洞见：

- **v1 不分轴**：把 mms 里会 balk 的 user 侧**写**动作（grant/toggle）也按「至多一次」跳过 → 没落地的修复永不重发 → 死循环（mms 19→12）。
- **v2 按 agent/user 分**：dump 发现 `check_payment_request`、`make_payment` 其实都是 **user 侧**，被 agent 门一并排除 → overdue 退回 0/4。
- **v3 按只读/写分**（复用现成 `_READONLY_TOOL_RE` / `_MUT_TOOL_RE`）：只读门（when 永不翻、查不改状态）需 at-most-once 跳过以推进到终末写；mutating 写会被用户 balk、必须保留重发。两头都对 → 0.975。

> **教训**：lever 的正确区分轴是行为语义（只读 vs 会 balk 的写），不是表面归属（agent/user）。

---

## 2 · 攻克最后残差 overdue_bill — 两层死锁

纯 B 唯一未解任务（service，0/4）。dump 4 个 trial 终局 task_state 实锤（`bill_paid=False, pay_allowed=True, make_payment 从没被调`），定位**两层叠加**，须 ELICIT + SEQ 齐治（数据见 HTML §2）：

- **第一层**：`makePayment` 门控 `pay_allowed==true`，授权藏在 task_instructions、schema 恒填 null → 永不激活。**L-ELICIT 解**。
- **第二层**：`check_payment_request`(只读)的 when 永不翻 → `_next_action` 循环选它、选不到 `make_payment`；且 check 输出被 8B 幻觉解析成 bill_paid=False → 子任务提前 done。**L-SEQ 解**。

结果：`B 0/4 → B+ELICIT 0/4 → B+ELICIT+SEQ 4/4`，telecom_small service 家族满分 24/24。

---

## 3 · strat90 泛化 — 极难复合集

strat90 = 分层抽的 90 任务（每家族 30），29/30 多故障、最深 11 故障同坏。参考线 solo 模式 @100 步 = 0.689。数据见 HTML §3。

两条泛化结论：

1. **「复合任务 × 步数」假说坐实**：user 模式 @100 步从 40 步的 0.20 跃到 0.578（B）/ 0.644（B+E+S），同 100 步下 solo→user 的纯模式鸿沟从天堑收窄到 **~4.5pp**（0.689 vs 0.644）。
2. **lever 正泛化、非过拟合**：ELICIT+SEQ 的 +6 任务**几乎全落在 service（25→29/30）**，与机制预测严丝合缝。

---

## 4 · mms 瓶颈诊断 — 两种截然不同的 regime

strat90 仅 mms 7/30 拖后腿。诊断揭示它和 telecom_small 的 mms 是**完全不同的问题**（对比/断崖数据见 HTML §4）：

| | telecom_small mms（已解 23/24） | strat90 mms（未解 7/30） |
|---|---|---|
| **本质** | 单故障 + 主诉不可判别根因 | 8–11 故障全栈复合（连通+数据+mms 层叠加） |
| **难点** | 选不对那一个修复 | 修复链太深 × 每修复开销 > 预算/耐心 |
| **解法** | SHOTGUN 盲修全试（绕过判别） | 未解 → 需吞吐 lever |

**机制（证据充分，是吞吐问题不是根因模糊）**：失败的 23 个 mms 任务 **23/23 都已恢复连通**、10/23 距完成只差 ≤2 个 mms 条件——REPLAN 已按因果序把该修的都试了。但每修复 = 诊断→指令→(balk 重发)→改→retest，一个 trace 里光 retest（`run_speed_test×5 + can_send_mms×5`）就吃十几步；× 8–11 故障 → 撞 100 步（9 例 max_steps）或 Hard 用户多轮 balk 后放弃（14 例 user_stop）。AND-gate 的 `can_send_mms` 零部分分 → 修好 9/11 = reward 0。故障数 9+ 时 pass^1 = 0/10（断崖）。

---

## 5 · 下一步方向

**主攻：复合 mms 吞吐 lever。** 把 SHOTGUN 的「批量 retest + 免逐项诊断」从单故障扩到全链——连通层修完测一次 speed、mms 层修完测一次 can_send_mms（而非每修必测），复合模式跳过逐项 `check_*`。直接砍每修复开销 → 更多修复挤进 100 步预算 + 减少用户轮数（缓解 balk 放弃）。

- 注意：user 设备动作不能批（一次一动作铁律），省的是**诊断 + retest 开销**，非合并操作。
- gate 需收回到 `service_status==connected` 后才启，避免跳过连通层修复。

**次要 / 配套：**

- 补 telecom_small 的 B+E+S 确认跑，坐实 0.975（去 ±0.05 噪声；overdue +4 是确定性结构修复，mms/mobile 小波动是噪声）。
- pass^4 衰减成因分析：已有 token/上下文日志显示失败 trial 的对话上下文（peak_prompt_exec）与回合数系统性高于成功 trial，可深挖长对话衰减。
