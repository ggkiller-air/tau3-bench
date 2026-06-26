# SchemaFlex 操作手册

## 1. 启动 vLLM

```bash
bash scripts/start_2gpu_server.sh --list-models        # 看可用模型
export CUDA_VISIBLE_DEVICES=6,7
bash scripts/start_2gpu_server.sh --model qwen3-8b  --port 9000   # 起服务 (默认开 prefix caching)
```

## 2. 跑 SchemaFlex（本地 8B）

`scripts/run.sh` 一条命令搞定：token 记录 + 该 run 缓存命中快照 + 跑完自动出报告。

```bash
cd /root/Project/SchemaFlex/tau2-bench
NUM_TRIALS=1 \
scripts/run.sh telecom_strat90  --task-set-name telecom_strat90  --max-concurrency 4
```

**常用参数**
  --task-set-name telecom_full   # 指定任务集
  --max-concurrency N             # 并发数
  --max-steps N                   # 最大步数；run.sh 也可用 MAX_STEPS= 覆盖
  --num-trials N                  # 每个任务跑几次；run.sh 推荐用 NUM_TRIALS=
  --agent-llm MODEL               # agent 模型；run.sh 推荐用 AGENT_LLM=
  --verbose-logs                  # 保存更详细日志

## 3. 跑 API 代理（如 gpt-5.4，做对比 baseline）

同一个脚本，加 `ENV_FILE=.env`（base/key 从 .env 读）+ `AGENT_LLM=` 换模型即可，自动跳过本地缓存快照：

```bash
ENV_FILE=.env AGENT_LLM=openai/gpt-5.4 \
scripts/run.sh gpt54/v1 --task-set-name telecom_full --num-tasks 30 --max-concurrency 3
```
（代理无本地缓存 → 报告 cache 显示 n/a；token/$ 仍出，但 $ 用的是占位单价，要算 gpt 真实成本就 `python3 scripts/report.py data/simulations/gpt54/v1 --price-in <真价> --price-out <真价>` 重出报告。）

## 4. 后处理：生成报告

```bash
python3 scripts/report.py data/simulations/<run>          # → <run>/report.html
```

## 5. 跑 user 模式（`schema_user_agent` + LLM 用户）

**user 模式**（agent 多轮引导 LLM-user 在手机上操作，工具非对称）用 `scripts/runUser.sh`。
双模型：**agent** = 本地 vLLM 小模型（`--agent-llm-args` 钉住 base），**user-sim** = `gpt-5.4`（base/key 从 `.env` 读）。

```bash
cd /root/Project/SchemaFlex/tau2-bench
# 先按第 1 节起好本地 vLLM（默认期望 qwen3-8b @ http://127.0.0.1:9000/v1）

# baseline（所有 lever 关 = 忠实静态走查）
scripts/runUser.sh user_base --task-set-name telecom_small --num-trials 4

# 当前最佳栈（telecom_small pass^1 0.975 / pass^4 0.90）：B + ELICIT + SEQ
#   B = REPLAN + CLOSE(常开) + GROUND
SCHEMAFLEX_REPLAN=1 SCHEMAFLEX_GROUND=1 SCHEMAFLEX_ELICIT=1 SCHEMAFLEX_SEQ=1 \
scripts/runUser.sh user_full --task-set-name telecom_small --num-trials 4
```

**Lever 开关**（环境变量，默认全关；置 `1` 开启。消融数据见 `stage_report.md`）

主线栈（REPLAN → CLOSE → GROUND → ELICIT → SEQ）：
  SCHEMAFLEX_REPLAN=1     # 因果链(produces/consumes)动态重排子任务，修复先行 + 剪死分支
  SCHEMAFLEX_GROUND=1     # 硬化 agent↔user 指令接缝（一次一动作、字面拼 tool/args、禁美化命名、检查类只 LOOK&report）
  SCHEMAFLEX_ELICIT=1     # give-up 前问用户 yes/no 补 null 上下文字段（pay_allowed/is_abroad）→ 治 overdue_bill 第一层
  SCHEMAFLEX_SEQ=1        # 只读门→写 的序列死锁：跳过已跑只读门推进到终末写 + 末位写未跑不判 done → 治 overdue_bill 第二层
  # L-CLOSE（success_when_all 满足即主动收尾；护栏：必须真 dispatch 过 mutating 工具）常开，无开关

replan-失败修复栈（strat90_replan 60/90 诊断；solo-schema 用到 user-mode 的接缝缺陷，默认关）：
  SCHEMAFLEX_PROV=1      # action 级 provenance：goal 字段(speed_test/can_send_mms)只能由其观测工具(run_speed_test/can_send_mms)那次回话写 → 禁 FIX 步乐观早写 → 灭 phantom 早闭 latch（治 mobile_data verify-stall）
  SCHEMAFLEX_UNLATCH=1   # proactive 早闭后用户仍在说话(没 STOP)=不认账 → 清 resolved+goal 字段、重开 VERIFY 恢复走查（PROV 兜底）
  SCHEMAFLEX_WATCHDOG=1  # episode 级护栏：同指令连发≥REPEAT(默3) → escalate（治 mms 逐字死锁）；连 STALL(默15) 轮无 goal-值变化/子任务完成 → escalate（止损 REPLAN-thrash/深度坍塌，省 $）
  SCHEMAFLEX_REPLY=1     # 把用户上一句注入 _phrase_instruction（balk 后据其诉求改口、给字面值）+ 把 app 名字面 grounding 从 shotgun-only 提到普通走查（治字面 app 名僵局）
  # 可调：SCHEMAFLEX_WATCHDOG_REPEAT（默3）/ SCHEMAFLEX_WATCHDOG_STALL（默15）
  # 注意：WATCHDOG 经实测在 strat90 上纯负（0 恢复/5 回归，escalate-loop），主线建议不开

高低协同（DyLight 稀疏纠错；离线诊断：8B StateUpdater 抽取是瓶颈，高后果区 ~19% 有后果错、且是"自信地错"故 SAGE 采样不确定性失效）：
  SCHEMAFLEX_HILO=1      # 载荷抽取（Tier1 终末 VERIFY/RETEST + mms 修复区；Tier2 别处由廉价结构门标记 8B 明显错）→ 大模型重抽并覆盖。8B 仍跑 Executor 措辞 + 多数抽取，主力不变；token sidecar 拆 updater vs updater_hi 出 $
  # 可调：SCHEMAFLEX_HILO_LLM（默 openai/gpt-5.4，走 .env 代理 base）/ SCHEMAFLEX_HILO_BASE（默 ""=OPENAI_API_BASE）
  #       SCHEMAFLEX_HILO_MAX（每集大模型纠错预算，默12）/ SCHEMAFLEX_HILO_VERIFY（Tier1 子任务集，逗号分隔，可改窄降 $）

实验性 / 已摘除（默认关，仅 A/B 消融用）：
  SCHEMAFLEX_SHOTGUN=1    # mms 盲修全试；40 步时代的突破 lever，放开到 100 步后反成负担、已从主线摘除（见 stage_report 关键转折一）
  SCHEMAFLEX_SAGE=1       # 结构化不确定性 stop/elicit 门（可调 SCHEMAFLEX_SAGE_LAMBDA/ALPHA/EPS/MAX_ASKS）
  SCHEMAFLEX_SUPERVISOR=1 # 卡死 junction 处稀疏大模型监督（SCHEMAFLEX_SUPERVISOR_LLM 默认 gpt-5.4、SCHEMAFLEX_SUPERVISOR_MAX 默认 4）
  SCHEMAFLEX_DIAG=1       # 打断 REPLAN 在 mms 上的错修抖动（需配合 REPLAN 一起开）

**可覆盖的 env**（同 run.sh 风格）
  NUM_TRIALS=4            # pass^4
  MAX_STEPS=150           # 默认 100
  AGENT_LLM=openai/xxx    # agent 小模型（默认 openai/qwen3-8b）
  USER_LLM=openai/xxx     # user-sim（默认 openai/gpt-5.4）
  VLLM_BASE=http://127.0.0.1:7000/v1   # agent 的本地 vLLM base（默认 9000）
  ENV_FILE=.env          # user-sim 的 base/key 来源
  TAU_USER=dummy_user    # 极少用：换掉 LLM 用户（注意是 TAU_USER，不是 USER——USER 是 shell 内置变量）

`--task-set-name` / `--max-concurrency` / `--num-tasks` 等照常透传给 `tau2 run`。
跑完自动出 `report.html`；家族/pass^k 对比另用 `python analyze_arm.py <run_dir> <baseline_dir>`。
