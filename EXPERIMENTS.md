# 实验请求 — Claude→Codex 工作流

> **协作约定**：Claude(规划)在本文件写实验请求(矩阵/命令/期望/报告格式)；**Codex(执行)** 照着跑实验，把结果写进 `EXPERIMENT_REPORT.md`(每轮追加一节)；Claude 读报告决定下一步。
> 项目背景见 `/mnt/ssd2/xll/CLAUDE.md`。散文中文，命令/路径原样。

---

## 环境(每轮跑前确认)

- conda env：跑 tau2 用 **`tau3bench`**；起 8B 用 **`vllm`**。
- **agent 后端 = 本地 qwen3-8b @ `http://127.0.0.1:9000/v1`**。跑前确认在线：
  ```bash
  curl -s --max-time 5 http://127.0.0.1:9000/v1/models
  ```
  若没在线：`cd /mnt/ssd2/xll/tau3-bench && bash start_vllm.sh`（GPU0；起绿配方见 CLAUDE.md §6），等 `/v1/models` 返回再跑。
- user-sim = gpt-5.4 @ micuapi（key/base 在 repo 根 `.env`，脚本会自动 source；本地代理已在脚本里 `no_proxy='*'` 处理）。
- 所有 `run_*.sh` 自带：source .env、关代理、`rm -rf` 旧产物、设 STATE_LOG/TOKEN_LOG/TOKEN_TRACE。**直接 `bash run_xxx.sh` 即可，不用手设环境变量。**
- 跑法：后台跑 + 等 `done rc=`：
  ```bash
  nohup bash run_xxx.sh > data/simulations/<SAVE>/run.log 2>&1 &
  until grep -q 'done rc=' data/simulations/<SAVE>/run.log; do sleep 30; done
  ```
- 每个 full 实验 = 20 任务 × 4 trial = 80 sims，~30-60 分钟。可串行跑（共享 GPU0，别并发多个 tau2 run）。

## 分析工具(纯读产物，repo 根)

- `python analyze_arm.py <dir> [baseline_dir]` → pass^k / 家族 / A1-A2-B 失败桶 / 终止 / 事件 / token。给了 baseline 则打 DELTA。
- `python analyze_tokens.py <dir>` → pass^4 衰减视角：成功 vs 失败 trial 的上下文(peak_prompt_exec)/回合数/token 均值对比 + 同任务 pass/fail 混合(衰减源)逐 trial 对比。
- `python dump_sim.py <save_name> <sim_id>` → 单 sim 全轨迹（排查某任务为何挂）。

---

## Round 1（2026-06-22）：100 步重测 + SHOTGUN/ELICIT 消融

**背景**：max_steps 已从 40 提到 **100**（telecom_small 脚本已改）。40 步下最佳栈 pass^1≈0.80/pass^4≈0.70，残差 apn(40步墙)/overdue_bill。**目标 pass^1≥0.90、pass^4≥0.80**。本轮要回答：①100 步下最佳栈的 pass^1/pass^4 落在哪；②SHOTGUN 在 100 步还值不值（正常走查有了余量，盲修优势可能缩水）；③残差 apn/overdue_bill 在 100 步是否自己就过了；④pass^4 在 100 步是否仍明显衰减（决定要不要深挖成因）。

### 实验矩阵（全部 telecom_small / num-trials 4 / max_steps 100，脚本已配好）

| 臂 | 栈 | 脚本 | 产物目录 |
|---|---|---|---|
| **A** | REPLAN+CLOSE+GROUND+SHOTGUN（当前最佳） | `run_replan_ground_shotgun_full.sh` | `user_replan_ground_shotgun_full` |
| **B** | REPLAN+CLOSE+GROUND（SHOTGUN 关，消融基线） | `run_replan_ground_full.sh` | `user_replan_ground_full` |
| **C** | A + ELICIT | `run_replan_ground_shotgun_elicit_full.sh` | `user_replan_ground_shotgun_elicit_full` |

> 串行跑 A、B、C。每个跑完先 `grep -c infrastructure_error <dir>/results.json`（应≈0；非 0 说明 micuapi 抖动，记下数量、必要时重跑该臂）。

### 每臂分析（跑完后）

```bash
# 总分 + 消融 delta（A、C 都对 B 打 delta）
python analyze_arm.py data/simulations/user_replan_ground_shotgun_full data/simulations/user_replan_ground_full
python analyze_arm.py data/simulations/user_replan_ground_shotgun_elicit_full data/simulations/user_replan_ground_shotgun_full
python analyze_arm.py data/simulations/user_replan_ground_full
# pass^4 衰减 token/上下文
python analyze_tokens.py data/simulations/user_replan_ground_shotgun_full
```

### 要在报告里回答的问题

1. **三臂的 pass^1 / pass^2 / pass^3 / pass^4**（4-trial）。
2. **逐家族 pass^1**（service / mobile_data / mms）。
3. **残差任务**：`break_apn_mms_setting`、`overdue_bill_suspension` 在各臂的 4-trial 通过数（0-4/4）——100 步下转正了吗？
4. **SHOTGUN 增量**：A vs B 的 pass^1/pass^4 delta——100 步下 SHOTGUN 还正吗？
5. **ELICIT 增量**：C vs A——overdue_bill 有没有动？有没有 derail（A2/A1 桶变化）？
6. **pass^4 衰减**：A 的 pass^1 与 pass^4 差多少？`analyze_tokens` 里失败 trial 的 peak_prompt_exec/回合数是否系统性高于成功 trial？（这决定要不要深挖衰减成因）

### 报告格式（写进 `EXPERIMENT_REPORT.md`，新开一节 `## Round 1 报告`）

- 一张三臂 × pass^1-4 的表 + 逐家族表。
- 残差任务 4-trial 通过数表（A/B/C）。
- `analyze_arm` 的 DELTA 原文（A-vs-B、C-vs-A）+ `analyze_tokens` 的原文输出。
- 3-5 行结论：SHOTGUN/ELICIT 在 100 步的去留、残差是否解决、pass^4 是否仍衰减、建议下一步。
- 任何异常（infra 错、某臂跑崩、infra 重跑）如实记。

---

## 历史轮次报告

见 `EXPERIMENT_REPORT.md`。
