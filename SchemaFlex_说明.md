# SchemaFlex 论文说明

> 统一命名 **SchemaFlex**(弃用 DyLight 这一名字)。
> 本文档汇总:① 系统逻辑细节;② 论文三大创新点;③ Introduction 立论草稿。
> 配套:实验现状/数据见 `PAPER_FINDINGS.md`;主表设计见本文 §4。

---

## 0. 一句话定位

在长程多轮工具交互任务中,**把"下一步做什么"的规划负担从模型手里拿走、交给确定性的 Schema 走查**;让**小模型(SLM)只做"自然语 ↔ 结构化状态"的逐步翻译 + 执行已锁定的单一动作**;只在**最易出错、最高后果的解析点**让大模型稀疏兜底。由此在长程任务上以 **~1/10 的大模型成本逼近大模型精度**,同时阻断多轮交互中的错误累积。

载体:**τ²-bench(Sierra)telecom/retail/airline** 三域,user 模式(工具非对称、一次一动作、reward 二值、撞 max_steps 判 0)。

---

## 1. 系统逻辑细节

### 1.1 三层架构

| 层 | 是谁 | 职责 | 用 LLM? |
|---|---|---|---|
| **控制层** | 纯 Python(Schema 走查) | 选子任务、何时收尾/放弃、动态重排、各 guard | ❌ 只路由,不理解内容 |
| **主力层** | 本地 **qwen3-8b**(SLM) | 每步内容活(3 角色) | ✅ 小模型,~90% token |
| **稀疏纠错层** | **gpt-5.4**(verify-offload) | 只复核/重抽 8B 最易错的载荷 | ✅ 大模型,~9–10% token |

> 关键设计原则:**大模型只纠"解析",从不碰"决策"**。让大模型管决策的三套尝试(SUPERVISOR / SELECT / SAGE)均 derail、全部弃用——这是 verify-offload 成立的反面对照。

代码落点:
- 主 agent:`src/tau2/agent/schema_agent.py`(`SchemaSoloAgent`)、`src/tau2/agent/schema_user_agent.py`(`SchemaUserAgent`,继承前者 + 对话路由 + 高低协同)。
- 注册:`src/tau2/registry.py` → `schema_solo_agent` / `schema_user_agent` / `llm_agent_solo_sanitized`(裸 ReAct baseline)。

### 1.2 Schema 与确定性控制层(创新#1 的"稳定性"来源)

- **Schema 工件**:`memory/{task_spec.json, subtask_spec.json}`(手写;可用 `SCHEMAFLEX_SPEC_DIR` 覆盖)。
  - `task_spec`:每 family 一条 `base_subtask_sequence`(诊断/修复子任务序列)+ `task_state_schema`(字段 + enums)+ `init_rules` + `success_when_all`(目标谓词)。
  - `subtask_spec`:每个子任务原语的 `base_actions`(动作空间 + `when` 条件)、`parse_rules`、`done_when_all`、`patch_ops_policy.allowed`(白名单)、`on_enter_reset`、`retest_guard`、`executor_sys_rules`。
- **规模**:3 family(`no_service` / `mobile_data` / `mms`)+ ~40 子任务原语(telecom)。family 路由靠任务 id 前缀(`_FAMILY_BY_PREFIX`)。
- **走查机制(通用、无任务特定 Python、无 eval / 无 regex)**:
  - `eval_pred`(`schema_agent.py:150`)——微型 DSL,只支持 `task_state.<f> ==|!= <lit>` 原子 + `and`/`or`,**不是 Python eval**。
  - `_active_subtask`(`:664`)——**sticky + guarded 走查**:已激活未完成的子任务粘住;否则按 `when` 顺序取第一个成立的。
  - `_apply_patch_ops`(`:207`)——**白名单强制**:只应用 `patch_ops_policy.allowed` 里的 `set` 操作 + enum 校验,其余静默丢弃。
  - `on_enter_reset`(`:650`)/ `retest_guard`(`:687`)——进入修复子任务时作废过期诊断字段;修复施加后只留 retest 工具,防 flip 工具二次触发。
- **核心思想**:eligible 子任务集 = 固定序列里 `when` 成立的子集,**运行时不增不减**。"卡权限就修"= schema 里早已存在的 `FIX_STORAGE_PERM` 因 `mms_storage_perm != true` 而**激活**,不是动态新增。→ 把"下一步做什么"离线编译进 schema,运行时只做确定性激活 + 排序 + 收尾时机。

### 1.3 SLM 三角色(LightWM:8B 维护一个全局结构化 task_state)

- **MacroStateInitializer**(每 episode 1 次):ticket → 初始 task_state。
- **Executor**(每步 1 次):在当前子任务**允许的工具子集**内,生成 ONE 动作 / 一条给用户的自然语指令(device 动作是用户工具,8B 不能直接调)。**一子任务一工具**(防 8B loop/fixate)。
- **StateUpdater**(每条结果/回话 1 次):把工具结果或客户自由文本解析成 `patch_ops`(更新 task_state)。
- **task_state**:扁平 dict,每步整体作为 JSON 传给三角色;由 `_apply_patch_ops` 原地更新。**没有 per-subtask 持久 local memory**——每子任务只有静态 spec,executor 每步临时拼 = 静态 spec + 全局 task_state。

### 1.4 Verify-Offload 高低协同(创新#1 的"verify-offload")

8B 的 **StateUpdater 解析是精度瓶颈**:高后果区 ~19% 抽取**有后果地错**(把 "can't send MMS" 写成 service_status、从失败回话读出 resolved、漏抽同句 co-field…),且**自信地错**(自洽采样只逮 13% → 基于采样方差的不确定性方法在此失效)。

**机制(两层路由,按后果而非置信;`schema_user_agent.py:1457`)**:8B 先解析每条 →
- **Tier-1(恒上大模型)**:终末验证 `VERIFY_SPEED/MMS/SERVICE`、`RETEST_MMS` + mms 修复区 `FIX_STORAGE_PERM/SMS_PERM/NETWORK_MODE_MMS/WIFI_CALLING`——错一次就 phantom-close 或漏解决。
- **Tier-2(别处只在可疑时)**:廉价结构门(写了 resolved / 纯提问却写 / 富信息回话却空写 / 失败回话却写成功值)标记 → 才上大模型。
- 预算 `SCHEMAFLEX_HILO_MAX`(默认 12);实测 ~3.9 次/任务、占 agent token ~9.4%。命中后**大模型输出整体覆盖 8B**。

### 1.5 动态调度 / planner(创新#3)

**纯 Python 的任务编排算出来 + LLM 稀疏纠错。** 把 solo 的干净流程适配到 user 模式的模糊真实对话:
- **REPLAN**(`SCHEMAFLEX_REPLAN`):按因果链(produces/consumes)动态重排 eligible 候选,修复先行、剪死分支(4 档 tier 排序)。
- **THRU**(`SCHEMAFLEX_THRU`):family 级 retest 在有 informative 诊断 pending 时降 tier,让诊断 + 修复先批量跑、AND-gate 末尾测一次(只动 family retest 排序,不碰承重的 per-fix retest)。
- **PROV**(`SCHEMAFLEX_PROV`):action 级 provenance,goal 字段只能由其观测工具那次回话写 → 堵 phantom resolved latch。
- 其余 lever(GROUND/ELICIT/SEQ/CLOSE/UNLATCH/REPLY)详见 `PAPER_FINDINGS.md §3`。

### 1.6 SchemaGen(创新#2:自动 schema 生成)

目录 `/root/Project/SchemaFlex/SchemaGen/`,流水线:
1. `gen_ir.py`(调 GPT-5.5):领域文档 + 工具源码 → IR(workflow_ir / tools_ir)。
2. `compile.py`(**确定性、无 LLM**):IR → `task_spec/subtask_spec`,实现 **C(骨架)+ P(结构不变量)+ M(8B 脚手架)**。
3. `inject_k.py`:注入 **K 层**(`k_caveats.json`,13 条手写领域感知:signal≠speed、MMS 否定、sentinel…)。
4. 闭环:`distill_runlog.py`(失败蒸馏)→ `analyze_runlog.py`(调 GPT 诊断 → repairs)→ 回灌 compile。
5. `diff.py`:对手写 golden 评 parity(guards 81/81、allow-list 40/40、done 36/40)。

**现状边界(诚实)**:
- 自动化程度 = **C+P+M 自动 + K 层种子手写(可经 run-log 闭环扩充)**,**不宣称零先验**。
- 只在 **solo 模式**验证过:strat90 hand 100% / auto 90%。**user 模式 parity 尚未补**。
- 只有 `telecom_data`,**retail/airline 尚无**(需手写,SchemaGen 后续蒸馏)。
- 生成产物结构与手写 spec 等价,可直接丢进 `SCHEMAFLEX_SPEC_DIR` 跑。

---

## 2. 论文三大创新点

| # | 创新点 | 实现 | 主证据 |
|---|---|---|---|
| **1** | **工具调用场景的 Schema-SLM 范式 + verify-offload**:确定性 schema 走查让 SLM 在长程任务上稳定;最易错的终末验证/载荷抽取 offload 给稀疏大模型 | §1.2–1.4 | user 模式 strat90:纯 Python 0.707 → +offload 0.818;大模型 token ~10× 少 |
| **2** | **SchemaGen 低工作量迁移 / 泛化鲁棒**:从领域文档(非 golden、非轨迹)自动编译结构等价 schema;反过来蒸馏手写经验 | §1.6 | solo parity hand 100% / auto 90%;+ retail/airline 迁移 |
| **3** | **动态调度 planner**:纯 Python 因果编排 + LLM 稀疏纠错,处理真实场景(user 模式模糊对话)的任务问题 | §1.5 | solo→user 提升;REPLAN/THRU 消融;LLM-planner derail 负结果作对照 |

**贯穿全文的一条核心论断**:
> 不是"高低模型按决策分工",而是 **"规划外置给确定性 schema → SLM 受约束地逐步翻译 → 大模型只稀疏纠正解析"**。三者各守边界,才能同时**降本(\$)**与**阻断错误累积(稳定性)**。

---

## 3. Introduction 立论(草稿)

**3.1 问题**。长程多轮工具交互(以 τ²-bench telecom user 模式为载体:工具非对称——agent 只有账户/计费工具,所有 device 动作是用户工具,排障 = 多轮指导用户 + 解读模糊自然语;一次一动作;reward 二值;撞 max_steps 判 0)。

**3.2 双重瓶颈**。
- **大模型 Agent**:高 token 开销 + 长上下文导致的**累积幻觉**——长程轨迹里每步小错沿对话放大。
- **小模型 Agent**:推理弱 + 上下文过载,**无法独立承担长程编排**——表现为 (1−p)^k 随步数 k 的成功率坍塌。
- → **成本 vs. 可靠性的传统权衡**。

**3.3 关键洞察(本文立论)**。长程任务中"下一步做什么"的规划负担,可以**离线编译进确定性 Schema**;SLM 只需做"自然语 ↔ 结构化状态"的逐步翻译并执行已锁定的单一动作;大模型**只在最易错、最高后果的解析点稀疏介入**。三者各守边界 → 同时降本 + 阻断错误累积。

**3.4 贡献(4 条)**。
1. **SchemaFlex 框架**:确定性 schema 走查 + SLM 三角色 + 稀疏大模型 verify-offload 的三层协同范式。
2. **verify-offload**:大模型按**后果**(而非置信/采样不确定性)路由,**只纠解析不碰决策**;附"大模型管决策必 derail"的对照负结果。
3. **动态调度 planner**:零额外大模型决策的确定性因果重排,把流程适配到真实模糊对话。
4. **SchemaGen**:从领域文档自动编译结构等价 schema,低工作量迁移到新域(C+P+M 自动 + 薄 K 层)。

**3.5 结果预告**。strat90(user 模式,深复合):纯 Python 栈 0.707 → +verify-offload 0.818(@100 步)/ 0.944(@150 步);大模型 token ↓ ~10.6×,verify-offload 仅占 ~9.4% token。标准三域主表:**匹配前沿精度,前沿 \$ 成本 ~1/10**(详见 §4)。

> ⚠️ Intro 红线:**不承诺"更高精度"**(标准 telecom 前沿已 ~0.98);**不宣称"fewer tokens"**(3 角色重发近静态 prompt,raw token 可能更多)。卖点是 **\$ 效率 + 长程稳定性 + 可迁移**。

---

## 4. 主表设计(FROZEN,详见 memory: project-paper-experiment-design)

- **域(4)**:telecom(**官方集 114,非我们自造的 2285**)/ retail 114 / airline 50 / +1 bench(待定)。
- **系统(5)**:`LLM-A`、`LLM-B`(榜单前沿,**仅抄成功率**)、`LLM-C`(成本锚,**自跑全指标**)、`SLM-ReAct`(同款 8B 裸 ReAct)、`SLM-Ours`(SchemaFlex)。
- **指标(4)**:pass^1 & pass^4;耗时 = 墙钟(硬件脚注)+ #turns(硬件无关);**Cost\$ + 前沿 token 占比(效率 headline)**;ETU(有效非缓存 token,辅助列)。
- **诚实边界**:① 2285/strat90 → "长程深复合压力测试" companion,非榜单口径;② τ² 官方榜只公布 pass^k 无 token/cost → 必须自跑 ≥1 前沿才有 ~1/10 成本对比;③ 绝不宣称 fewer tokens;④ 耗时 raw 秒不可跨系统直接比。
- **关键路径(挡 retail/airline 两行)**:手写 retail/airline schema + 去掉 agent prompt 的 "telecom" 硬编码 + 选定 LLM-A/B/C + 定 +1 bench。
- **Companion**:C1 深复合压力(strat90/2285,步数墙 @100→@150);C2 消融(verify-offload on/off、planner on/off、super-additivity、LLM-planner derail 负结果);第二张表(SchemaGen 泛化)。

---

## 5. 指标公式

- `pass^k = avg_task C(c,k)/C(n,k)`,n = num-trials = 4。
- `ETU = prompt_tokens × (1 − cache_hit) + completion_tokens`。
- `Cost$ = Σ_calls ( eff_input × price_in + output × price_out )`,各模型按公开列价(8B 用 DeepInfra/Together qwen3-8b 档;前沿用列价;pricing 表带日期)。

---

*生成日期 2026-06-27。系统逻辑引用的 file:line 基于当时代码;实际写作/引用前请核对仍存在。*
