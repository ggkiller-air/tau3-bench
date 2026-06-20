#!/bin/bash
# 本地 Qwen3-8B 推理后端（tau3-bench 的 agent llm），GPU0，端口 9000。
# 在 tmux 会话里跑，脱离 Claude 工具生命周期，长存活。
cd /mnt/ssd2/xll/tau3-bench
LOG=vllm_qwen3-8b_9000.log
exec env CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/ssd2/jhn/ECAgent-project/models/Qwen3-8B/Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --host 127.0.0.1 --port 9000 \
  --max-model-len 24576 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 --enable-chunked-prefill \
  2>&1 | tee "$LOG"
