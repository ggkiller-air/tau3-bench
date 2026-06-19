# SchemaFlex 操作手册

## 1. 启动 vLLM

```bash
bash scripts/start_2gpu_server.sh --list-models        # 看可用模型
export CUDA_VISIBLE_DEVICES=5,6
bash scripts/start_2gpu_server.sh --model qwen3-8b  --port 7000   # 起服务 (默认开 prefix caching)
```

## 2. 跑 SchemaFlex（本地 8B）

`scripts/run.sh` 一条命令搞定：token 记录 + 该 run 缓存命中快照 + 跑完自动出报告。

```bash
cd /root/Project/SchemaFlex/tau2-bench
NUM_TRIALS=4 \
scripts/run.sh GPT_GenSchema --task-set-name telecom_full  --max-concurrency 4
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
