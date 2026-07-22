# SchemaFlex 现状文档 · Lever / 实现 / 实验结果


## 1. 任务设定

- **τ³-bench = tau2-bench v1.0.0 telecom domain**。本工作聚焦 **user 模式**：
  - **工具非对称**：agent 只有账户/计费工具；**所有 device 动作（toggle_roaming / run_speed_test / grant_app_permission / reset_apn …共 59 个）都是用户工具，agent 调不到**。排障 = agent 多轮**指导 LLM-user 操作手机 + 解读模糊自然语回复**。
  - **一次只能一个动作**；**reward 二值**（最终 DB/env 状态匹配，无部分分）；**撞 max_steps 直接判 0**。
- **任务集**：`telecom_strat90`（90 任务，极深复合，29/30 多故障、均 ~3.8 故障、最深 9-11 故障）；`telecom_full`（**2285 任务**，**87% 是 mms**、60% ≥5 故障，是全量评测集）；`telecom_small`（20，**非本工作目标**）。

---

## 2. 架构与实现方法

**三层：确定性 Python 控制层 + 8B 主力（每步内容）+ gpt-5.4 稀疏纠错（仅最易错处）。** 外加 gpt-5.4 扮的用户（属环境，非 agent）。

| 层 | 是谁 | 干什么 | LLM？ |
|---|---|---|---|
| **控制层** | 纯 Python（schema 走查） | 选子任务、何时收尾/放弃、REPLAN 重排、各 guard | ❌ 只路由不理解内容 |
| **主力** | 本地 **qwen3-8b** @127.0.0.1:9000 | 每步内容活（3 角色） | ✅ 小模型，~90% token |
| **稀疏纠错** | **gpt-5.4**（HILO，.env 代理） | 只复核/重做 8B 最易错的载荷抽取 | ✅ 大模型，~9-10% token |

### 2.1 底座（LightWM/ESSA）：8B 三角色维护一个全局结构化 `task_state`
- **MacroInitializer**（每 episode 1 次）：ticket → 初始 task_state。
- **Executor**（每步 1 次）：在当前子任务允许的工具内，生成**一条给用户的自然语指令**（device 动作是用户工具，8B 不能直接调）。
- **StateUpdater**（每条用户回话 1 次）：把客户自由文本解析成 `patch_ops`（更新 task_state 字段）。
- **没有 per-subtask 持久 local memory**；每子任务只有静态 spec（base_actions = 该子任务 action space / parse_rules / done_when_all）。executor 每步临时拼 = 静态 spec + 全局 task_state。

### 2.2 Schema 编排
- schema：每 family 一条 `base_subtask_sequence`（诊断/修复子任务序列）+ `when` 门控 + `success_when_all` 目标谓词 + 每子任务的 `done_when_all`。
- **eligible 集 = 固定序列里 when 对 task_state 成立的子集**——子任务/action 运行时**不增不减**。「卡权限就修」= 早就在 schema 里的 fixStoragePerm 因 `mms_storage_perm!=true` 成立而**激活**，不是动态新增。
- **离线编译进 schema，运行时只做确定性激活 + 排序 + 收尾时机**。这是"小模型也能胜任"的关键：把"下一步做什么"从 8B 手里拿走、交给确定性的 schema 走查，8B 只做「自然语↔task_state 互译 + 执行已锁定的单一动作」，**从不答"下一步做什么"**。

### 2.3 高低协同：稀疏大模型纠错
- **动机**：8B 的 **StateUpdater 解析是瓶颈**——高后果区 ~19% 抽取**有后果地错**（把"can't send MMS"写成 service_status、从失败回话读出 resolved、漏抽同句 co-field…），且**自信地错**：8B 自洽采样只逮 13%（SAGE 式"不确定性=采样方差"在此失效）。
- **机制（两层路由，按后果而非置信）**：8B 先解析每条 →
  - **Tier-1（总上大模型）**：终末验证 `VERIFY_SPEED/VERIFY_MMS/VERIFY_SERVICE/RETEST_MMS` + mms 修复区 `FIX_STORAGE_PERM/FIX_SMS_PERM/FIX_NETWORK_MODE_MMS/FIX_WIFI_CALLING`——错一次就 phantom-close 或漏解决，大模型直接重抽覆盖。
  - **Tier-2（别处只在可疑时）**：廉价结构门（写了 resolved / 从纯提问幻觉 / 富信息回话却空写 / 失败回话却写成功值）标记 → 才上大模型。
  - 每集预算 `SCHEMAFLEX_HILO_MAX=12`；实测 ~3.9 次/任务、占 agent token 的 ~9.4%。
- **关键洞察**：大模型只纠**解析（StateUpdater）**、**不碰决策（Executor）**——这是它不 derail 的根本。proactive 让大模型管决策的 SUPERVISOR/SELECT/SAGE 三次都失败、全弃。

---

## 3. Lever 清单

| Lever | flag | 治什么 | 机制 | 层 |
|---|---|---|---|---|
| **REPLAN** | `SCHEMAFLEX_REPLAN` | 静态走查把诊断/修复顺序写死、烧步数 | 按因果链（produces/consumes）动态重排 eligible 候选，修复先行、剪死分支；4 档 tier 排序 | Python | 
| **GROUND** | `SCHEMAFLEX_GROUND` | agent↔user 指令接缝糙（M2） | 一次一动作、字面拼 tool/args、禁美化命名（messaging≠Messages）、检查类只 LOOK&report、丢 goal_template/Step-N 脚手架 | Python（约束 8B 输出） |
| **ELICIT** | `SCHEMAFLEX_ELICIT` | 缺只有用户知道的上下文前置字段 | give-up 前问一句 yes/no 补 null 字段（pay_allowed/is_abroad），判据=无人 produce 的 ticket-context 字段 | Python+8B |
| **SEQ** | `SCHEMAFLEX_SEQ` | 「只读门→mutating 写」序列死锁 | R-skip：跳过已跑的**只读**非末位 action；R-doneguard：末位是 mutating 写且未 dispatch 时 withhold done。**区分轴=只读 vs 写** | Python |
| **CLOSE** | 常开 | 目标已达却无人收尾→撞 max_steps | success_when_all 满足即主动置 resolved 收尾；护栏=必须真 dispatch 过 mutating 工具 | Python |
| **PROV** | `SCHEMAFLEX_PROV` | goal 字段被 FIX 步乐观早写→phantom resolved latch | **action 级 provenance**：speed_test/can_send_mms 只能由其**观测工具**（run_speed_test/can_send_mms）那次回话写 | Python |
| **UNLATCH** | `SCHEMAFLEX_UNLATCH` | resolved 误锁死冻结 | 早闭后用户仍说话→清 resolved 重开验证（PROV 开后基本无 phantom 可解，冗余无害） | Python | 
| **REPLY** | `SCHEMAFLEX_REPLY` | 字面 app 名僵局 / 指令不贴用户上一句 | 把用户上一句注入指令措辞 + 普通走查钉死字面 "messaging" | Python+8B |
| **HILO** | `SCHEMAFLEX_HILO` | 8B StateUpdater 解析错（精度瓶颈） | 稀疏大模型按后果路由重抽载荷（见 §2.3） | 8B+大模型 |
| **THRU** | `SCHEMAFLEX_THRU`(+`_THRU_V1`) | 深复合 mms 的 family 级 retest 空转、烧步数 | family 级 verifyMms/retestMms（产 goal-core、恒 eligible 抢 tier-3）在有 informative 诊断 pending 时**降 tier-2**，让诊断+修复先批量跑、AND-gate 末尾测一次。**只动 family retest 排序，不碰承重的 per-fix retest** | Python |

**已否决（写进 limitations/ablation 的负结果）**：
- **L-FRT / DEFER**（砍每个 FIX_* 内嵌的 per-fix retest）：**两人独立验证净有害**——多trial strat90 −0.152。教训：**深长对话里 per-fix retest 是承重的**（给交互打拍子、防终末验证被 stale 值满足、维持用户参与），不能删。
- **L-STEP**（balk 时升级 step-by-step 指令）：strat90 上未带来稳定增益、且误伤 service（scoping）；REPLY 已覆盖其核心。
- **SUPERVISOR / SELECT / SAGE / DIAG / WATCHDOG**：让大模型管**决策**或基于采样不确定性→derail / 噪声级，全弃。**正面教训：大模型只该纠解析、不该碰决策**（HILO 成立、这些失败的对照）。

---

## 4. 实验结果

> 主集 = `telecom_strat90`（90，极深复合）。agent=本地 qwen3-8b；user-sim & HILO=gpt-5.4。max-conc 4（实测稳）、max-errors 200/500。

### 4.1 主结果：schema 编排 + 8B 在 strat90（user 模式）

| 栈 | pass^1 | 口径 | 说明 |
|---|---:|---|---|
| solo 模式参考（agent 自己点工具，非 user 模式） | 0.689 | 单trial,100步 | 不可比基准（无 LLM-user/无一次一动作） |
| **THRU-v1 单独**（REPLAN+GROUND+ELICIT+SEQ+CLOSE+THRU） | **0.707** | 3-trial | 纯 Python 编排，无大模型纠错 |
| **HILO 单独**（wzh 栈：+PROV+UNLATCH+REPLY+HILO，无 THRU） | **0.756** | 单trial（我们机器复现；wzh 报 0.78） | 加大模型稀疏纠错 |
| **HILO + THRU @100 步** | **0.818** | **3-trial 稳定**（270 sims） | 合并栈 |
| **HILO + THRU @150 步** | **0.944** | 单trial | 放开步数 |

- **逐家族（HILO+THRU @100，3-trial）**：service **90/90(1.0)**、mobile_data 80/90(0.889)、mms 51/90(0.567)。
- **pass^k 衰减（HILO+THRU @100）**：pass^1 0.818 / pass^2 0.756 / pass^3 0.711（~0.05/k；深 mms 跨 trial flaky，是稳定性成本）。

### 4.2 两个机制结论（消融/分析）

**(a) 协同 = 超加性**：THRU 单独 0.707、HILO 单独 0.756 → 合起来 **0.818**（> 任一单项）。两个**正交杠杆**：HILO 提每步正确率 p、THRU 降步数 k，各补对方够不到的。增量成本：THRU 纯 Python **零额外 LLM**；HILO ~9.4% token。

**(b) 步数墙是主残余瓶颈，非"用户耐心墙"**：HILO+THRU 把 max_steps 100→150：

| | @100 | @150 | |
|---|---|---|---|
| pass^1 | 0.833(单trial) | **0.944** | +0.111 |
| mms | 17/30 | **27/30** | +10 |
| max_steps 失败 | 11 | **0** | 全消失 |
| A1_repair_no_close | 10 | **0** | 全消失 |
| 残余失败 | 15 | **5（真 user_stop）** | |

→ 深 mms 残差**主要是步数不够（修了没收尾），不是 OOS**。THRU 把会话压短，150 步就够且不 OOS.

**(c) PROV 的贡献**（wzh 实测）：mobile_data 21→29/30（堵 phantom resolved latch），是 mobile_data 满分的主因。

### 4.3 成本/效率（实测，写"小模型省钱"卖点用）

- **HILO 大模型 token（updater_hi）**：strat90 ~2.1K/任务、full@150 ~3.8K/任务；占 agent token ~9.4%。
- **单 trial strat90 总 gpt-5.4 ≈ 5.5M token（其中 user-sim ~4.84M 主导、HILO ~0.2M）**；官方 GPT-5 list 价 ≈ $7/trial，HILO 净增量仅 ~$0.4/trial。
- **效率主张**：相对全大模型 agent，大模型 token 少 ~10.6×（wzh 测）；不是"近免费"（8B 自托管≈免费，$ 几乎都在那 ~9.4% 稀疏大模型）。可主张：**"8B + ~9.4%-token 的稀疏大模型监督，逼近大模型精度、只花 ~1/10 的大模型成本"**。

---